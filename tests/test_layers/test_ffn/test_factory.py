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
from dl_techniques.layers.ffn.factory import (
    FFN_REGISTRY,
    STRICT_DROPPED_KEY_MARKER,
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


def _strictness_break(fn) -> Optional[str]:
    """Run ``fn``; return the strict-factory dropped-key message, or ``None``.

    THE VACUITY TRAP THIS EXISTS TO AVOID: at step 5 this was a
    ``logging.Handler`` on the **`dl`** logger, because ``create_ffn_layer``
    WARNED about a key it had to drop. Step 6 turned that warning into a raise
    (D-023). A warning recorder left in place would then capture nothing, ever
    -- every site in the sweep below would report a clean zero, the suite would
    stay green, and the guard would be measuring absolutely nothing. Changing
    the instrument was mandatory, not cosmetic.

    Any raise that is NOT a dropped key -- a missing required param, a rank
    mismatch -- was already loud BEFORE the flip, so it is not something
    strictness newly broke and is deliberately not reported here.

    :param fn: A zero-argument callable that reaches an FFN construction.
    :return: The ``ValueError`` message if ``fn`` failed on a dropped caller
        key, else ``None`` (both for success and for any other exception).
    :rtype: Optional[str]
    """
    try:
        fn()
    except Exception as exc:  # noqa: BLE001 - classification is the point
        message = str(exc)
        return message if STRICT_DROPPED_KEY_MARKER in message else None
    return None


# --- the hand-written builders -------------------------------------------
# One entry per FILE returned by `_derive_ffn_construction_files()`. Each
# builder must actually REACH the `create_ffn_layer` call: a builder that
# raises before construction reports zero drops and is vacuous, which is what
# `test_every_builder_actually_reaches_its_ffn` exists to catch.
#
# DECISION plan-2026-07-30T140922-8af1028f/D-020
# Every builder uses the site's OWN default `ffn_type` (or the literal the site
# hardcodes). This sweep's job is the PRODUCTION-REACHABLE surface, which is
# what the strict factory can break today -- it is deliberately NOT a 21-type
# grid.
#
# It was RIGHT not to be one at step 5, when 4 sites --
# `layers/time_series/mixed_sequential_block.py`, `models/nam/cell.py`,
# `models/sam/image_encoder.py` and `models/tree_transformer/components.py` --
# passed `activation` unconditionally and dropped it for `differential`,
# `gelu_tanh`, `squared_relu` and `swiglu`. All four adopted
# `assemble_ffn_config` at step 6 (D-021/D-022) and now measure 0/21; per-site
# grids live in the owning test suites. Turning THIS sweep into a grid would
# duplicate them and couple a `src/`-wide coverage guard to every layer's
# construction preconditions.

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

    This was the GATE that unblocked making `create_ffn_layer` raise. It is now
    the REGRESSION guard for that raise: each builder constructs the real
    owning layer/model at ITS OWN defaults, and a site that hands the factory a
    key its `ffn_type` does not accept is a hard construction failure, not a
    log record.
    """

    def test_the_strictness_classifier_bites(self) -> None:
        """RED-proof: a `None` from a blind classifier proves nothing.

        Injects a deliberately misspelled key and requires the classifier to
        see it, then runs the identical call WITHOUT the injection as a
        control. Both halves are load-bearing: the first shows it recognises a
        real strictness failure, the second shows it is not simply
        always-firing.
        """
        injected = _strictness_break(
            lambda: create_ffn_layer(
                'mlp', hidden_dim=8, output_dim=8, hiden_dim=512
            )
        )
        assert injected is not None and 'hiden_dim' in injected, (
            f"_strictness_break returned {injected!r} for a deliberately "
            f"misspelled 'hiden_dim'. It is BLIND, so every None reported by "
            f"the sweep below would be vacuous."
        )
        control = _strictness_break(
            lambda: create_ffn_layer('mlp', hidden_dim=8, output_dim=8)
        )
        assert control is None, (
            f"the classifier fired {control!r} on a clean call; it cannot "
            f"distinguish a drop from a non-drop."
        )

    def test_the_strictness_classifier_ignores_other_raises(self) -> None:
        """A raise that predates strictness must NOT be reported as a break.

        Without this, `_strictness_break` could simply return `str(exc)` for
        every exception, and the grid-style guards elsewhere in the suite would
        report false breaks for every type that legitimately raises on a
        missing required parameter.
        """
        other = _strictness_break(lambda: create_ffn_layer('mlp'))
        assert other is None, (
            f"a MISSING-required-param raise was classified as a strictness "
            f"break ({other!r}); the classifier is too wide."
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
        broke = _strictness_break(_FFN_CONSTRUCTION_SITE_BUILDERS[site])
        assert broke is None, (
            f"src/dl_techniques/{site} hands create_ffn_layer a key its "
            f"ffn_type does not accept, which is now a HARD construction "
            f"failure: {broke}. Fix the site (correct the parameter name, or "
            f"route its generic conveniences through assemble_ffn_config) -- "
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


# =====================================================================
# STEP 6 -- the strict factory (plan-2026-07-30T140922-8af1028f, D-023)
# =====================================================================


def _minimal_required_ffn_args(ffn_type: str) -> Dict[str, Any]:
    """Smallest kwargs covering `ffn_type`'s `required_params`.

    Construction only -- no `build()` -- so the widths need not be compatible
    with any real input shape; they only need to pass `validate_ffn_config`.
    """
    sized = {
        'hidden_dim': 16, 'output_dim': 8, 'units': 8,
        'features': 8, 'filters': 8,
    }
    return {
        p: sized.get(p, 8) for p in FFN_REGISTRY[ffn_type]['required_params']
    }


class TestStrictDroppedKeyRaise:
    """`create_ffn_layer` RAISES on a caller key the ffn_type cannot accept.

    Four guards, one per way this change can go wrong:

    1. a caller-supplied unknown key RAISES, naming the key;
    2. a registry `optional_params` DEFAULT never raises -- the wide-predicate
       regression guard, RED-proved against `set(kwargs) - required_params`;
    3. `validate_ffn_config`'s required-params raise still fires FIRST, with
       its message unchanged (I-3);
    4. a wrapper layer's `.keras` round-trip still loads, value-exact and with
       an equal weight count (assumption A-5, which had only ever been
       REASONED from reading the file).
    """

    # --- guard 1 ---------------------------------------------------------
    def test_caller_supplied_unknown_key_raises_naming_the_key(self) -> None:
        with pytest.raises(ValueError, match="definitely_not_an_mlp_param"):
            create_ffn_layer(
                'mlp', hidden_dim=8, output_dim=8,
                definitely_not_an_mlp_param=1,
            )

    def test_the_raise_names_the_ffn_type_and_the_accepted_set(self) -> None:
        """The message must be ACTIONABLE, not merely present.

        A reader must be able to tell "I typed the wrong name" from "I chose
        the wrong ffn_type" without opening the factory, so the message has to
        carry the type AND the parameters it does accept.
        """
        with pytest.raises(ValueError) as excinfo:
            create_ffn_layer('mlp', hidden_dim=8, output_dim=8, hiden_dim=512)
        message = str(excinfo.value)
        assert 'hiden_dim' in message
        assert "'mlp'" in message
        # one real mlp parameter, quoted from the registry rather than typed
        assert 'activation' in message
        assert STRICT_DROPPED_KEY_MARKER in message

    def test_a_valid_key_is_still_accepted(self) -> None:
        """CONTROL: strictness must not reject a key the type DOES take."""
        layer = create_ffn_layer(
            'mlp', hidden_dim=8, output_dim=8, activation='relu'
        )
        assert layer is not None

    # --- guard 2: the wide-predicate regression guard ---------------------
    @pytest.mark.parametrize('ffn_type', sorted(FFN_REGISTRY))
    def test_a_registry_optional_default_never_raises(self, ffn_type) -> None:
        """The factory's OWN defaults must not trip its own strictness.

        RED-PROOF (executed 2026-07-30, by exact-string edit in the shipped
        factory -- never `git stash`/`git checkout`): replacing the predicate's
        right-hand side with the too-narrow

            dropped = sorted(set(kwargs) - set(ffn_info['required_params']))

        fired this test for **21/21** types. That is the widening that actually
        breaks: it makes every OPTIONAL parameter an error.

        RECORDED NEGATIVE RESULT, because the plan predicted the opposite:
        substituting `set(params) - valid_param_names` (the "merged dict"
        variant D-002 called unworkable) changes NOTHING -- 205/205 green.
        `params` is `optional_params` updated with `kwargs`, and
        `optional_params`'s keys are a subset of `valid_param_names` by
        construction, so the two expressions are extensionally EQUAL. The
        `kwargs` form is still the right one to ship (see the anchor), but it
        is not load-bearing for the reason the plan gave, and a RED proof
        built on that reason would have quietly proved nothing.
        """
        entry = FFN_REGISTRY[ffn_type]
        # Every registry default, handed back to the factory explicitly. These
        # are by construction accepted keys, so none may be reported as dropped.
        #
        # The required params are supplied TOO, and that is load-bearing: with
        # only the optional ones, `validate_ffn_config` raises first for 15 of
        # the 21 types, `_strictness_break` correctly returns None, and this
        # test passes without ever reaching the predicate it exists to guard.
        # (Measured during the RED proof below: 21/21 "passed" vacuously.)
        kwargs = dict(entry['optional_params'])
        kwargs.update(_minimal_required_ffn_args(ffn_type))
        assert set(entry['required_params']) <= set(kwargs)
        broke = _strictness_break(lambda: create_ffn_layer(ffn_type, **kwargs))
        assert broke is None, (
            f"create_ffn_layer({ffn_type!r}) reported its OWN registry "
            f"optional_params as unsupported: {broke}. The dropped-key "
            f"predicate has been widened to read the merged `params` dict; "
            f"revert it to `set(kwargs) - valid_param_names`."
        )

    def test_optional_defaults_pass_while_a_typo_raises(self) -> None:
        """Both halves of the predicate, together, on one type.

        A registry optional default passes while a caller typo raises. A
        too-wide predicate fails the first half; a deleted predicate fails the
        second. Neither half alone can see both failures.
        """
        defaults = dict(FFN_REGISTRY['mlp']['optional_params'])
        assert defaults, "mlp has no optional_params; this test measures nothing"
        create_ffn_layer('mlp', hidden_dim=8, output_dim=8, **defaults)
        with pytest.raises(ValueError, match='not_a_real_key'):
            create_ffn_layer(
                'mlp', hidden_dim=8, output_dim=8, not_a_real_key=1, **defaults
            )

    # --- guard 3: I-3, the pre-existing required-params raise -------------
    def test_required_params_raise_still_fires_first_and_unchanged(self) -> None:
        """I-3: `validate_ffn_config` runs BEFORE the drop check, unchanged.

        Both a missing REQUIRED key and an unsupported EXTRA key are present in
        the same call. The required-params message must be the one that
        surfaces, in its pre-existing wording -- if strictness ever moved ahead
        of validation, a caller who simply forgot a required parameter would be
        told about their extra key instead.
        """
        with pytest.raises(ValueError) as excinfo:
            create_ffn_layer('mlp', hidden_dim=8, bogus_extra_key=1)
        message = str(excinfo.value)
        assert "Required parameters missing for mlp: ['output_dim']" in message, (
            f"the pre-existing required-params message changed or was "
            f"pre-empted by the strictness raise: {message}"
        )
        assert STRICT_DROPPED_KEY_MARKER not in message, (
            "the strictness raise fired before validate_ffn_config; the order "
            "in create_ffn_layer must stay validate-then-filter (I-3)."
        )

    # --- guard 4: A-5, probed rather than reasoned ------------------------
    @pytest.mark.parametrize(
        'ffn_type', ['mlp', 'swiglu', 'differential', 'geglu', 'glu']
    )
    def test_keras_round_trip_survives_the_strict_factory(
            self, ffn_type) -> None:
        """A-5, PROBED: a strict raise cannot break `.keras` LOADING.

        The plan REASONED this from reading the file -- `from_config`
        reconstructs from already-resolved params rather than from
        `ffn_type` + kwargs -- and never executed it. This executes it, on a
        wrapper layer inside a real `keras.Model`, for five types whose
        registry entries each carry 7-9 `optional_params` (so the merged-dict
        hazard is genuinely present).

        Weight COUNT is asserted, not just output shape: a shape-only
        round-trip once passed on a model that restored ZERO weights.
        """
        from dl_techniques.layers.transformers.transformer import TransformerLayer

        keras.utils.set_random_seed(1234)
        inputs = keras.Input(shape=(6, 16))
        outputs = TransformerLayer(
            hidden_size=16, num_heads=2, intermediate_size=32,
            ffn_type=ffn_type, activation='relu', name='blk',
        )(inputs)
        model = keras.Model(inputs, outputs)

        x = np.random.RandomState(0).randn(3, 6, 16).astype('float32')
        before = np.array(model(x, training=False))
        n_before = len(model.weights)
        assert n_before > 0

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'strict_round_trip.keras')
            model.save(path)
            loaded = keras.models.load_model(path)

        after = np.array(loaded(x, training=False))
        assert len(loaded.weights) == n_before, (
            f"{ffn_type}: weight COUNT changed across the round-trip "
            f"({n_before} -> {len(loaded.weights)}); a shape-equal output does "
            f"not prove the weights came back."
        )
        np.testing.assert_array_equal(before, after)


class TestFormerlyDroppingConstructionSites:
    """The 3 sites that adopted `assemble_ffn_config` with DISCARD semantics.

    Step 5 measured four sites passing `activation` unconditionally, dropping
    it for `differential`, `gelu_tanh`, `squared_relu` and `swiglu` (4/21 each,
    except `mixed_sequential_block.py` at 1/21). The fourth,
    `layers/time_series/mixed_sequential_block.py`, chose RENAME semantics for
    `differential` and is guarded in its own suite, because a dropped-key grid
    is structurally blind to a missing rename.

    These three chose DISCARD (D-022) -- none of them ever enumerated FFN
    types, so their `activation` is an unconditional default rather than an
    expressed per-type intent. This class pins BOTH halves of that choice: the
    21-type grid (no strictness break anywhere) AND the discard itself, which
    the grid alone cannot see.

    The grid lives HERE rather than in each model's own suite so it shares the
    one `_strictness_break` classifier, whose bite is RED-proved above.
    """

    @staticmethod
    def _nam(ffn_type: str):
        from dl_techniques.models.nam.cell import NAMCell, NAMConfig
        cell = NAMCell(NAMConfig(
            hidden_size=16, num_heads=2, intermediate_size=32,
            ffn_type=ffn_type, hidden_act='relu',
        ))
        cell.build((None, 8, 16))
        return cell.ffn

    @staticmethod
    def _sam(ffn_type: str):
        from dl_techniques.models.sam.image_encoder import ViTBlock
        block = ViTBlock(dim=16, num_heads=2, ffn_type=ffn_type,
                         activation='relu')
        block.build((None, 8, 8, 16))
        return block.ffn

    @staticmethod
    def _tree(ffn_type: str):
        from dl_techniques.models.tree_transformer.components import (
            TreeTransformerBlock,
        )
        block = TreeTransformerBlock(
            hidden_size=16, num_heads=2, intermediate_size=32,
            ffn_type=ffn_type, hidden_act='relu',
        )
        block.build(((None, 8, 16), (None, 8), (None, 8, 8)))
        return block.ffn

    @property
    def _sites(self):
        return {'nam': self._nam, 'sam': self._sam, 'tree': self._tree}

    #: The 4 types each of these sites dropped `activation` for, measured at
    #: HEAD before step 6 with a handler on the `dl` logger.
    FORMERLY_DROPPED_TYPES = ('differential', 'gelu_tanh', 'squared_relu',
                              'swiglu')

    @pytest.mark.parametrize('site', ['nam', 'sam', 'tree'])
    @pytest.mark.parametrize('ffn_type', sorted(FFN_REGISTRY))
    def test_no_ffn_type_is_broken_by_strictness(self, site, ffn_type) -> None:
        build = self._sites[site]
        broke = _strictness_break(lambda: build(ffn_type))
        assert broke is None, (
            f"{site} with ffn_type={ffn_type!r} hands create_ffn_layer a key "
            f"that type does not accept: {broke}"
        )

    @pytest.mark.parametrize('site', ['nam', 'sam', 'tree'])
    def test_the_formerly_dropped_types_actually_construct(self, site) -> None:
        """Anti-vacuity: a zero also comes from a site that raises EARLIER.

        These four are exactly the cells the fix addressed, so if the adoption
        had instead broken their construction with a non-strictness error, the
        grid above would still be green.
        """
        build = self._sites[site]
        for ffn_type in self.FORMERLY_DROPPED_TYPES:
            assert build(ffn_type) is not None, (
                f"{site} did not construct an FFN for {ffn_type!r}"
            )

    @pytest.mark.parametrize('site', ['nam', 'sam', 'tree'])
    def test_differential_keeps_its_default_branch_activation(
            self, site) -> None:
        """D-022 DISCARD, stated as an assertion rather than only in prose.

        These sites deliberately do NOT rename their generic activation onto
        `DifferentialFFN.branch_activation`, because they never expressed a
        per-type intent for it -- contrast D-021. If a future change routes it
        through `build_transformer_ffn_config` (which owns that rename), this
        test is the one that must be updated, deliberately, rather than a
        silent behaviour change nobody notices.
        """
        ffn = self._sites[site]('differential')
        assert keras.activations.serialize(ffn.branch_activation) == 'gelu'


# =========================================================================
# The `ffn_args` / `ffn_config` population -- the channel the sweep above
# structurally CANNOT see.
# =========================================================================
#
# `TestFFNConstructionSiteSweep` above derives its site list from `ast.Call`
# nodes naming `create_ffn_layer` / `create_ffn_from_config`, and builds each
# site at its OWN default `ffn_type`. Both choices are deliberate and both are
# blind to the defect class below:
#
#   * `models/qwen/qwen3.py` contains NO factory call at all. It reaches the
#     factory through `TransformerLayer(ffn_args=...)` and through
#     `MoEConfig(...ExpertConfig(ffn_config=...))`, so it is not in that file
#     set and never could be.
#   * `ffn_args` is the ONE channel `assemble_ffn_config` forwards UNFILTERED
#     (D-017), on purpose -- it is how an end user's typo stays visible to the
#     strict factory (D-023). A caller that builds that dict ITSELF therefore
#     hands the factory keys nobody typed, and the raise blames the user for
#     the model's own convenience defaults.
#
# That is precisely what shipped: at `3ada9cdb` `Qwen3(ffn_type='mlp')` raised
# `create_ffn_layer('mlp'): 1 unsupported parameter(s) ['ffn_expansion_factor']`.
# MEASURED across the full 21-type registry, HEAD vs a pristine `f013c232`
# worktree, on `Qwen3` / `Qwen3SOM` / `Qwen3MEGA`:
#
#     dense block path  (`ffn_args`)  : 12 of 21 constructed -> 1 of 21
#     MoE expert path (`ffn_config`)  :  7 of 21 constructed -> 1 of 21
#
# i.e. 11 of the 12 working `ffn_type` values died on three shipped model
# families, plus 6 more on the MoE path, plus `Qwen3Next`'s MoE path.
#
# Derivation (re-run to reproduce; nothing here is keyed on a line number):
#
#     grep -rn "ffn_args=\|encoder_ffn_args=\|ffn_config=" src/ --include=*.py
#
# which over-reports, because most of those hits are a LAYER forwarding its own
# `self.ffn_args` parameter downstream -- a pass-through, not a self-built dict.
# `_derive_self_built_ffn_kwarg_sites()` below separates the two structurally.

_SELF_BUILT_KWARG_NAMES = frozenset({"ffn_args", "encoder_ffn_args", "ffn_config"})
_PREFILTER_NAME = "assemble_ffn_config"

#: The site is a HAZARD when a self-built dict meets a NON-constant `ffn_type`:
#: the dict's keys are fixed at authoring time while the type is chosen by the
#: caller, so no set of keys can be right for every type.
_KIND_FILTERED = "FILTERED"
_KIND_RAW = "RAW"
_KIND_PASSTHROUGH = "PASSTHROUGH"
_DYNAMIC = "<dynamic>"


def _call_name(node) -> Optional[str]:
    """Name of the function a `ast.Call` invokes, attribute or bare."""
    func = node.func
    return getattr(func, "id", None) or getattr(func, "attr", None)


def _dict_keys(node) -> Optional[List[Any]]:
    """Constant keys of an `ast.Dict`, or ``None`` for a non-literal."""
    if not isinstance(node, _sweep_ast.Dict):
        return None
    return [k.value if isinstance(k, _sweep_ast.Constant) else None
            for k in node.keys]


def _derive_self_built_ffn_kwarg_sites() -> List[Dict[str, Any]]:
    """Every ``ffn_args=`` / ``ffn_config=`` argument built IN `src/`.

    Derived by AST, not grep, so a pass-through (``ffn_args=self.ffn_args``)
    is separated from a self-built dict (``ffn_args={...}``) structurally
    rather than by pattern-guessing. For each site it records:

    * ``kind`` -- ``FILTERED`` (the value is an ``assemble_ffn_config(...)``
      call), ``RAW`` (a dict literal, directly or via a local name), or
      ``PASSTHROUGH`` (anything else -- an attribute, a call, a ``.copy()``).
    * ``ffn_type`` -- the constant string chosen at that call site, or
      ``"<dynamic>"`` when it is an expression. Read from the sibling
      ``ffn_type=`` keyword, or from the dict's own ``"type"`` entry for the
      ``ffn_config`` form.
    * ``keys`` -- the dict's constant keys, for ``RAW`` sites.

    Only ``RAW`` sites can carry the defect; ``PASSTHROUGH`` sites hand on
    whatever their own caller supplied and are that caller's responsibility.

    :return: one record per site, sorted by (file, line).
    :rtype: List[Dict[str, Any]]
    """
    rows: List[Dict[str, Any]] = []
    for path in _SWEEP_SRC_ROOT.rglob("*.py"):
        rel = path.relative_to(_SWEEP_SRC_ROOT).as_posix()
        try:
            tree = _sweep_ast.parse(path.read_text())
        except SyntaxError:  # pragma: no cover - defensive
            continue
        # module-wide Name -> assigned values. Crude on purpose: a name bound to
        # a dict literal ANYWHERE in the module is treated as self-built, which
        # errs toward reporting a hazard rather than hiding one.
        assigns: Dict[str, List[Any]] = {}
        for node in _sweep_ast.walk(tree):
            if (isinstance(node, _sweep_ast.Assign)
                    and len(node.targets) == 1
                    and isinstance(node.targets[0], _sweep_ast.Name)):
                assigns.setdefault(node.targets[0].id, []).append(node.value)

        for node in _sweep_ast.walk(tree):
            if not isinstance(node, _sweep_ast.Call):
                continue
            for kw in node.keywords:
                if kw.arg not in _SELF_BUILT_KWARG_NAMES:
                    continue
                value = kw.value
                kind, keys, dict_node = _KIND_PASSTHROUGH, None, None
                prefiltered_type = None
                if (isinstance(value, _sweep_ast.Call)
                        and _call_name(value) == _PREFILTER_NAME):
                    kind = _KIND_FILTERED
                    # The pre-filter's FIRST positional argument IS the ffn_type
                    # this site targets. Read it here: a FILTERED site often has
                    # no sibling `ffn_type=` keyword at all (the MoE
                    # `ffn_config` form carries the type INSIDE the dict, which
                    # is now hidden behind the call), and without this the site
                    # would report `ffn_type=None` and drop out of
                    # `_derive_dynamic_type_model_classes` -- silently
                    # un-sweeping the model. MEASURED: that is exactly what
                    # happened to `models/qwen/qwen3_next.py`.
                    if value.args:
                        first = value.args[0]
                        prefiltered_type = (
                            first.value
                            if isinstance(first, _sweep_ast.Constant)
                            else _DYNAMIC
                        )
                elif isinstance(value, _sweep_ast.Dict):
                    kind, dict_node = _KIND_RAW, value
                elif isinstance(value, _sweep_ast.Name):
                    candidates = assigns.get(value.id, [])
                    if any(isinstance(c, _sweep_ast.Call)
                           and _call_name(c) == _PREFILTER_NAME
                           for c in candidates):
                        kind = _KIND_FILTERED
                    else:
                        literals = [c for c in candidates
                                    if isinstance(c, _sweep_ast.Dict)]
                        if literals:
                            kind, dict_node = _KIND_RAW, literals[0]
                if dict_node is not None:
                    keys = _dict_keys(dict_node)

                ffn_type = prefiltered_type
                for sibling in node.keywords:
                    if sibling.arg == "ffn_type":
                        ffn_type = (
                            sibling.value.value
                            if isinstance(sibling.value, _sweep_ast.Constant)
                            else _DYNAMIC
                        )
                if dict_node is not None and keys and "type" in keys:
                    for k, v in zip(dict_node.keys, dict_node.values):
                        if isinstance(k, _sweep_ast.Constant) and k.value == "type":
                            ffn_type = (v.value
                                        if isinstance(v, _sweep_ast.Constant)
                                        else _DYNAMIC)

                rows.append({
                    "file": rel,
                    "line": node.lineno,
                    "kwarg": kw.arg,
                    "kind": kind,
                    "ffn_type": ffn_type,
                    "keys": keys,
                })
    return sorted(rows, key=lambda r: (r["file"], r["line"]))


#: Minimal construction kwargs for each model class that owns a self-built FFN
#: kwarg dict AND exposes a caller-settable `ffn_type`. The CLASS SET is derived
#: (`_derive_dynamic_type_model_classes`); only "how do I instantiate this
#: cheaply" is hand-written, because that cannot be derived. A new model with
#: the same shape therefore fails `test_every_dynamic_type_model_has_a_recipe`
#: loudly instead of shipping the regression again.
#:
#: Two variants per class where the two paths differ: the dense block path feeds
#: `TransformerLayer(ffn_args=...)`, the MoE path feeds
#: `ExpertConfig(ffn_config=...)`. Only the second was ever exercised by
#: `Qwen3Next`, and neither was exercised by any test at `3ada9cdb`:
#: `grep -rn ffn_type tests/test_models/test_qwen/` returns 2 hits, both inside
#: a `get_config()` key list.
_MODEL_FFN_RECIPES: Dict[str, List[Dict[str, Any]]] = {
    "Qwen3": [
        {"vocab_size": 64, "hidden_size": 32, "num_layers": 1,
         "num_attention_heads": 4, "num_key_value_heads": 2, "max_seq_len": 8},
        {"vocab_size": 64, "hidden_size": 32, "num_layers": 1,
         "num_attention_heads": 4, "num_key_value_heads": 2, "max_seq_len": 8,
         "moe_layers": [0], "num_experts": 2, "num_experts_per_tok": 1,
         "moe_intermediate_size": 64},
    ],
    "Qwen3SOM": [
        {"vocab_size": 64, "hidden_size": 32, "num_layers": 1,
         "num_attention_heads": 4, "num_key_value_heads": 2, "max_seq_len": 8},
        {"vocab_size": 64, "hidden_size": 32, "num_layers": 1,
         "num_attention_heads": 4, "num_key_value_heads": 2, "max_seq_len": 8,
         "moe_layers": [0], "num_experts": 2, "num_experts_per_tok": 1,
         "moe_intermediate_size": 64},
    ],
    "Qwen3MEGA": [
        {"vocab_size": 64, "hidden_size": 32, "num_layers": 1,
         "num_attention_heads": 4, "num_key_value_heads": 2, "max_seq_len": 8},
        {"vocab_size": 64, "hidden_size": 32, "num_layers": 1,
         "num_attention_heads": 4, "num_key_value_heads": 2, "max_seq_len": 8,
         "moe_layers": [0], "num_experts": 2, "num_experts_per_tok": 1,
         "moe_intermediate_size": 64},
    ],
    "Qwen3Next": [
        # `num_experts > 1` is the MoE trigger here and its default is 64, so
        # this single recipe already exercises the `ffn_config` path.
        {"vocab_size": 64, "hidden_size": 32, "num_layers": 1,
         "num_attention_heads": 4, "num_key_value_heads": 2, "max_seq_len": 8,
         "num_experts": 2, "num_experts_per_tok": 1,
         "moe_intermediate_size": 64},
    ],
}


def _derive_dynamic_type_model_classes() -> Dict[str, Any]:
    """Model classes owning a self-built FFN kwarg dict at a dynamic `ffn_type`.

    Derived: take every site whose ``kind`` is ``RAW`` or ``FILTERED`` with
    ``ffn_type == "<dynamic>"`` in a module under ``models/``, import that
    module, and collect the ``keras.Model`` subclasses DEFINED there that take
    an ``ffn_type`` constructor parameter.

    :return: class name -> class object.
    :rtype: Dict[str, Any]
    """
    import importlib
    import inspect

    modules = {
        row["file"] for row in _derive_self_built_ffn_kwarg_sites()
        if row["file"].startswith("models/")
        and row["kind"] in (_KIND_RAW, _KIND_FILTERED)
        and row["ffn_type"] == _DYNAMIC
    }
    found: Dict[str, Any] = {}
    for rel in sorted(modules):
        dotted = "dl_techniques." + rel[:-len(".py")].replace("/", ".")
        module = importlib.import_module(dotted)
        for name, obj in vars(module).items():
            if (inspect.isclass(obj) and issubclass(obj, keras.Model)
                    and obj.__module__ == dotted
                    and "ffn_type" in inspect.signature(obj.__init__).parameters):
                found[name] = obj
    return found


class TestModelBuiltFFNKwargDictSweep:
    """A MODEL that builds its own FFN kwarg dict must pre-filter it.

    This is the population `TestFFNConstructionSiteSweep` cannot reach, and the
    one the strictness flip (D-023) actually broke. Two halves, because either
    alone is defeatable:

    * a STATIC rule over every self-built site in `src/` -- catches a new site
      the moment it is written, including in a model this file never imports;
    * an EXECUTED 21-type sweep over the model classes that own one -- catches
      the case where the static rule is satisfied in form (a pre-filter is
      called) but the wrong dict is filtered.
    """

    def test_the_derivation_is_not_blind(self) -> None:
        """Anti-vacuity floor: the AST walk must actually find sites.

        A silently-empty walk -- a renamed kwarg, a moved `src/` root, a
        `SyntaxError` swallowed by the `except` -- would make every assertion
        below pass over an empty set, forever.
        """
        rows = _derive_self_built_ffn_kwarg_sites()
        assert len(rows) >= 15, (
            f"only {len(rows)} `ffn_args=`/`ffn_config=` sites found under "
            f"{_SWEEP_SRC_ROOT}; the walk is blind"
        )
        kinds = {row["kind"] for row in rows}
        assert kinds >= {_KIND_RAW, _KIND_FILTERED, _KIND_PASSTHROUGH}, (
            f"the walk found only {sorted(kinds)}; it is no longer "
            f"distinguishing self-built dicts from pass-throughs, so the "
            f"hazard rule below cannot fire"
        )
        files = {row["file"] for row in rows}
        assert "models/qwen/qwen3.py" in files, (
            "the known-defective site `models/qwen/qwen3.py` is not in the "
            "derived inventory; the walk no longer covers the regression it "
            "was written for"
        )

    def test_no_raw_self_built_dict_meets_a_dynamic_ffn_type(self) -> None:
        """THE RULE. A fixed key set cannot be right for a caller-chosen type.

        # DECISION plan-2026-07-30T140922-8af1028f/D-037
        Do NOT relax this to "warn". It is the ONLY instrument in the repo that
        sees a caller reaching `create_ffn_layer` WITHOUT calling it, and its
        absence is exactly why `Qwen3(ffn_type='mlp')` shipped broken.
        """
        hazards = [
            row for row in _derive_self_built_ffn_kwarg_sites()
            if row["kind"] == _KIND_RAW and row["ffn_type"] == _DYNAMIC
        ]
        assert not hazards, (
            "self-built FFN kwarg dict handed to a CALLER-CHOSEN `ffn_type`:\n"
            + "\n".join(
                f"  {r['file']}:{r['line']} {r['kwarg']}={{{', '.join(map(str, r['keys'] or []))}}}"
                for r in hazards
            )
            + "\n\nThese keys are the MODEL's own conveniences, not a user's "
              "request, so wrap the dict in `assemble_ffn_config(<ffn_type>, "
              "{...})`. Leaving it raw sends them down the deliberately "
              "UNFILTERED `ffn_args` channel (D-017), where the strict factory "
              "(D-023) raises and blames the user for a key the model injected."
        )

    def test_every_constant_type_site_passes_only_keys_that_type_accepts(
            self) -> None:
        """A raw dict is fine when the site also pins `ffn_type` to a literal.

        Those sites (`models/clip/model.py` -> `swiglu`,
        `models/modern_bert/*` -> `geglu`) are checked against that exact
        type's registry entry instead, which is the strongest statement
        available without constructing the whole model.
        """
        offenders = []
        for row in _derive_self_built_ffn_kwarg_sites():
            if row["kind"] != _KIND_RAW or row["ffn_type"] in (None, _DYNAMIC):
                continue
            info = FFN_REGISTRY.get(row["ffn_type"])
            assert info is not None, (
                f"{row['file']}:{row['line']} names unregistered ffn_type "
                f"{row['ffn_type']!r}"
            )
            accepted = (set(info["required_params"])
                        | set(info["optional_params"])
                        | {"type", "name"})
            unsupported = sorted(set(row["keys"] or []) - accepted)
            if unsupported:
                offenders.append((row, unsupported))
        assert not offenders, "\n".join(
            f"{r['file']}:{r['line']} passes {bad} which "
            f"{r['ffn_type']!r} does not accept"
            for r, bad in offenders
        )

    def test_every_dynamic_type_model_has_a_recipe(self) -> None:
        """A NEW model with this shape must fail here, not in production."""
        derived = set(_derive_dynamic_type_model_classes())
        assert derived == set(_MODEL_FFN_RECIPES), (
            f"derived model classes {sorted(derived)} != recipes "
            f"{sorted(_MODEL_FFN_RECIPES)}. Add a construction recipe to "
            f"`_MODEL_FFN_RECIPES` (or remove a stale one) so the executed "
            f"sweep below actually covers it."
        )

    @pytest.mark.parametrize("class_name", sorted(_MODEL_FFN_RECIPES))
    def test_model_constructs_across_the_whole_registry(
            self, class_name) -> None:
        """EXECUTED: no registry type may fail with the strictness message.

        Types that fail for a DIFFERENT reason (a required parameter the model
        has no value for, e.g. `kan`'s `features`) were already loud before the
        flip and are deliberately tolerated -- `_strictness_break` returns
        `None` for them. The anti-vacuity floor is
        `test_the_sweep_actually_builds_something`.

        THE FORWARD PASS IS HONESTLY NOT YET LOAD-BEARING, and that is recorded
        rather than dressed up. It was added on the theory that `MoELayer`
        builds its `FFNExpert`s lazily and so would need data to flow; MEASURED
        against an injected raw `ffn_config` in `models/qwen/qwen3.py`, the
        `Qwen3` MoE path in fact raises at CONSTRUCTION, so a construction-only
        sweep catches every site that exists today. The `model(x)` call is kept
        anyway, at ~95s of the ~140s runtime, because "it builds" and "it runs"
        are different claims and the whole subject of this class is a sweep that
        was green over the wrong population. If a future site creates its FFN in
        `build()`, this is what will see it.
        """
        cls = _derive_dynamic_type_model_classes()[class_name]
        tokens = np.ones((1, 4), dtype="int32")
        broken = []
        for recipe in _MODEL_FFN_RECIPES[class_name]:
            for ffn_type in sorted(FFN_REGISTRY):
                def _build(t=ffn_type, r=recipe):
                    return cls(ffn_type=t, **r)(tokens)

                message = _strictness_break(_build)
                if message is not None:
                    broken.append((ffn_type, sorted(recipe), message))
        assert not broken, (
            f"{class_name} newly fails the strict factory for "
            f"{len(broken)} (ffn_type, recipe) cells:\n"
            + "\n".join(f"  {t}: {m}" for t, _, m in broken[:5])
        )

    @pytest.mark.parametrize("class_name", sorted(_MODEL_FFN_RECIPES))
    def test_the_sweep_actually_builds_something(self, class_name) -> None:
        """Anti-vacuity: `test_model_constructs_...` passes on a dead model.

        `_strictness_break` returns `None` for ANY non-strictness exception, so
        a model that raised on every single type -- a broken import, a renamed
        constructor argument, a forward pass that never reaches an FFN --
        would report zero breaks and look green. Pin that the default `swiglu`
        really constructs AND really runs, on every recipe.
        """
        cls = _derive_dynamic_type_model_classes()[class_name]
        tokens = np.ones((1, 4), dtype="int32")
        for recipe in _MODEL_FFN_RECIPES[class_name]:
            model = cls(ffn_type="swiglu", **recipe)
            assert model(tokens) is not None

    def test_a_genuine_end_user_ffn_args_typo_still_raises(self) -> None:
        """The fix must NOT have re-armed the silent drop it replaced.

        The model now pre-filters its OWN dict, but `TransformerLayer.ffn_args`
        remains the unfiltered end-user channel (D-017). A user's typo must
        still reach the factory and raise, naming the key.
        """
        from dl_techniques.layers.transformers import TransformerLayer

        with pytest.raises(ValueError) as excinfo:
            TransformerLayer(
                hidden_size=32, num_heads=4, intermediate_size=64,
                ffn_type='mlp', ffn_args={'hiden_dim': 512},
            )
        assert 'hiden_dim' in str(excinfo.value)
        assert STRICT_DROPPED_KEY_MARKER in str(excinfo.value)
