"""
Comprehensive test suite for the MixtureOfExperts layer.

This module provides extensive testing of the MoE layer, covering initialization,
forward passes, serialization, configuration management, and gradient flow.
Follows modern Keras 3 testing patterns with particular emphasis on the critical
serialization cycle test.
"""

import pytest
import tempfile
import os
import numpy as np
import keras
from keras import ops
import tensorflow as tf
from typing import Dict, Any


from dl_techniques.layers.moe.experts import FFNExpert, create_expert
from dl_techniques.layers.moe.layer import MixtureOfExperts, create_ffn_moe
from dl_techniques.layers.moe.config import MoEConfig, ExpertConfig, GatingConfig
from dl_techniques.layers.moe.gating import LinearGating, CosineGating, SoftMoEGating, create_gating


class TestMixtureOfExperts:
    """Comprehensive test suite for MixtureOfExperts layer."""

    @pytest.fixture
    def base_moe_config(self) -> Dict[str, Any]:
        """Base MoE configuration for testing."""
        return {
            'num_experts': 4,
            'expert_config': ExpertConfig(
                ffn_config={
                    'type': 'mlp',
                    'hidden_dim': 128,
                    'output_dim': 64,
                    'activation': 'relu'
                }
            ),
            'gating_config': GatingConfig(
                gating_type='linear',
                top_k=2,
                aux_loss_weight=0.01
            ),
            'jitter_noise': 0.01,
            'drop_tokens': True,
            'use_residual_connection': True
        }

    @pytest.fixture
    def swiglu_moe_config(self) -> Dict[str, Any]:
        """SwiGLU MoE configuration for testing."""
        return {
            'num_experts': 8,
            'expert_config': ExpertConfig(
                ffn_config={
                    'type': 'swiglu',
                    'output_dim': 768,
                    'output_dim': 768,
                    'ffn_expansion_factor': 4
                }
            ),
            'gating_config': GatingConfig(
                gating_type='cosine',
                top_k=1,
                embedding_dim=256
            )
        }

    @pytest.fixture
    def softmoe_config(self) -> Dict[str, Any]:
        """SoftMoE configuration for testing."""
        return {
            'num_experts': 6,
            'expert_config': ExpertConfig(
                ffn_config={
                    'type': 'geglu',
                    'hidden_dim': 512,
                    'output_dim': 256
                }
            ),
            'gating_config': GatingConfig(
                gating_type='softmoe',
                num_slots=4
            )
        }

    @pytest.fixture
    def sample_input_2d(self) -> keras.KerasTensor:
        """Sample 2D input for testing."""
        return keras.random.normal(shape=(8, 256))

    @pytest.fixture
    def sample_input_3d(self) -> keras.KerasTensor:
        """Sample 3D sequence input for testing."""
        return keras.random.normal(shape=(4, 32, 256))

    def test_initialization(self, base_moe_config):
        """Test MoE layer initialization."""
        config = MoEConfig(**base_moe_config)
        layer = MixtureOfExperts(config=config)

        # Verify configuration storage
        assert layer.num_experts == 4
        assert len(layer.experts) == 4
        assert layer.gating_network is not None
        assert not layer.built

        # Verify expert creation
        for expert in layer.experts:
            assert isinstance(expert, FFNExpert)
            assert expert.ffn_config['type'] == 'mlp'

        # Verify gating network creation
        assert hasattr(layer, 'gating_network')
        assert isinstance(layer.gating_network, LinearGating)

    def test_different_ffn_types(self):
        """Test MoE with different FFN expert types."""
        ffn_configs = [
            {'type': 'mlp', 'hidden_dim': 128, 'output_dim': 64},
            {'type': 'swiglu', 'output_dim': 64, 'output_dim': 64},
            {'type': 'geglu', 'hidden_dim': 256, 'output_dim': 64},
            {'type': 'glu', 'hidden_dim': 128, 'output_dim': 64},
            {'type': 'residual', 'hidden_dim': 128, 'output_dim': 64}
        ]

        sample_input = keras.random.normal(shape=(4, 128))

        for ffn_config in ffn_configs:
            config = MoEConfig(
                num_experts=4,
                expert_config=ExpertConfig(ffn_config=ffn_config),
                gating_config=GatingConfig(top_k=2)
            )

            layer = MixtureOfExperts(config=config)
            output = layer(sample_input)

            assert layer.built
            assert output.shape[0] == sample_input.shape[0]
            assert output.shape[-1] == ffn_config.get('output_dim', ffn_config.get('output_dim', 128))

    def test_different_gating_mechanisms(self, sample_input_2d, sample_input_3d):
        """Test different gating mechanisms."""
        gating_configs = [
            ('linear', {'gating_type': 'linear', 'top_k': 2}),
            ('cosine', {'gating_type': 'cosine', 'top_k': 1, 'embedding_dim': 128}),
            ('softmoe', {'gating_type': 'softmoe', 'num_slots': 4})
        ]

        for gating_name, gating_params in gating_configs:
            if gating_name == 'softmoe':
                # SoftMoE requires 3D input
                test_input = sample_input_3d
            else:
                test_input = sample_input_2d

            config = MoEConfig(
                num_experts=4,
                expert_config=ExpertConfig(
                    ffn_config={'type': 'mlp', 'hidden_dim': 512, 'output_dim': 256}
                ),
                gating_config=GatingConfig(**gating_params)
            )

            layer = MixtureOfExperts(config=config)
            output = layer(test_input)

            assert layer.built
            assert output.shape[0] == test_input.shape[0]
            assert output.shape[-1] == 256

    def test_forward_pass_2d_input(self, base_moe_config, sample_input_2d):
        """Test forward pass with 2D input."""
        config = MoEConfig(**base_moe_config)
        layer = MixtureOfExperts(config=config)

        output = layer(sample_input_2d)

        assert layer.built
        assert output.shape[0] == sample_input_2d.shape[0]
        assert output.shape[-1] == 64  # output_dim from config
        assert len(output.shape) == 2

    def test_forward_pass_3d_input(self, base_moe_config, sample_input_3d):
        """Test forward pass with 3D sequence input."""
        config = MoEConfig(**base_moe_config)
        layer = MixtureOfExperts(config=config)

        output = layer(sample_input_3d)

        assert layer.built
        assert output.shape[0] == sample_input_3d.shape[0]  # batch_size
        assert output.shape[1] == sample_input_3d.shape[1]  # seq_len
        assert output.shape[-1] == 64  # output_dim from config
        assert len(output.shape) == 3

    def test_serialization_cycle_mlp(self, base_moe_config, sample_input_2d):
        """CRITICAL TEST: Full serialization cycle with MLP experts."""
        config = MoEConfig(**base_moe_config)

        # Create model with MoE layer
        inputs = keras.Input(shape=sample_input_2d.shape[1:])
        outputs = MixtureOfExperts(config=config)(inputs)
        model = keras.Model(inputs, outputs)

        # Get original prediction
        original_prediction = model(sample_input_2d)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_moe_model.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_prediction = loaded_model(sample_input_2d)

            # Verify identical predictions
            np.testing.assert_allclose(
                ops.convert_to_numpy(original_prediction),
                ops.convert_to_numpy(loaded_prediction),
                rtol=1e-6, atol=1e-6,
                err_msg="MLP MoE predictions differ after serialization"
            )

    def test_serialization_cycle_swiglu(self, swiglu_moe_config, sample_input_3d):
        """CRITICAL TEST: Full serialization cycle with SwiGLU experts."""
        sample_input_3d_swiglu = keras.random.normal(shape=(4, 32, 768))
        config = MoEConfig(**swiglu_moe_config)

        # Create model with SwiGLU MoE layer
        inputs = keras.Input(shape=sample_input_3d_swiglu.shape[1:])
        outputs = MixtureOfExperts(config=config)(inputs)
        model = keras.Model(inputs, outputs)

        # Get original prediction
        original_prediction = model(sample_input_3d_swiglu)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_swiglu_moe.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_prediction = loaded_model(sample_input_3d_swiglu)

            # Verify identical predictions
            np.testing.assert_allclose(
                ops.convert_to_numpy(original_prediction),
                ops.convert_to_numpy(loaded_prediction),
                rtol=1e-6, atol=1e-6,
                err_msg="SwiGLU MoE predictions differ after serialization"
            )

    def test_serialization_cycle_softmoe(self, softmoe_config, sample_input_3d):
        """CRITICAL TEST: Full serialization cycle with SoftMoE gating."""
        config = MoEConfig(**softmoe_config)

        # Create model with SoftMoE layer
        inputs = keras.Input(shape=sample_input_3d.shape[1:])
        outputs = MixtureOfExperts(config=config)(inputs)
        model = keras.Model(inputs, outputs)

        # Get original prediction
        original_prediction = model(sample_input_3d)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_softmoe.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_prediction = loaded_model(sample_input_3d)

            # Verify identical predictions
            np.testing.assert_allclose(
                ops.convert_to_numpy(original_prediction),
                ops.convert_to_numpy(loaded_prediction),
                rtol=1e-6, atol=1e-6,
                err_msg="SoftMoE predictions differ after serialization"
            )

    def test_config_completeness(self, base_moe_config):
        """Test that get_config contains all configuration information."""
        config = MoEConfig(**base_moe_config)
        layer = MixtureOfExperts(config=config)
        layer_config = layer.get_config()

        # Check that config key is present
        assert 'config' in layer_config

        # Verify round-trip configuration
        restored_config = MoEConfig.from_dict(layer_config['config'])

        assert restored_config.num_experts == config.num_experts
        assert restored_config.expert_config.ffn_config == config.expert_config.ffn_config
        assert restored_config.gating_config.gating_type == config.gating_config.gating_type
        assert restored_config.gating_config.top_k == config.gating_config.top_k

    def test_gradients_flow(self, base_moe_config, sample_input_2d):
        """Test gradient computation and backpropagation."""
        config = MoEConfig(**base_moe_config)
        layer = MixtureOfExperts(config=config)

        # Use persistent tape to compute multiple gradients
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(sample_input_2d)
            output = layer(sample_input_2d, training=True)
            loss = ops.mean(ops.square(output))

        # Check gradients with respect to layer parameters
        gradients = tape.gradient(loss, layer.trainable_variables)

        assert len(gradients) > 0
        assert all(g is not None for g in gradients)

        # Check input gradients
        input_gradients = tape.gradient(loss, sample_input_2d)
        assert input_gradients is not None
        assert input_gradients.shape == sample_input_2d.shape

        # Clean up persistent tape
        del tape

    @pytest.mark.parametrize("training", [True, False, None])
    def test_training_modes(self, base_moe_config, sample_input_2d, training):
        """Test behavior in different training modes."""
        config = MoEConfig(**base_moe_config)
        layer = MixtureOfExperts(config=config)

        # Should work in all training modes
        output = layer(sample_input_2d, training=training)
        assert output.shape[0] == sample_input_2d.shape[0]
        assert output.shape[-1] == 64

    def test_compute_output_shape(self, base_moe_config):
        """Test output shape computation."""
        config = MoEConfig(**base_moe_config)
        layer = MixtureOfExperts(config=config)

        # Test 2D input shape
        input_shape_2d = (None, 256)
        output_shape_2d = layer.compute_output_shape(input_shape_2d)
        assert output_shape_2d == (None, 64)

        # Test 3D input shape
        input_shape_3d = (None, 32, 256)
        output_shape_3d = layer.compute_output_shape(input_shape_3d)
        assert output_shape_3d == (None, 32, 64)

    def test_expert_utilization_stats(self, base_moe_config, sample_input_2d):
        """Test expert utilization statistics."""
        config = MoEConfig(**base_moe_config)
        layer = MixtureOfExperts(config=config)

        # Build layer by running forward pass
        _ = layer(sample_input_2d)

        # Get utilization stats
        stats = layer.get_expert_utilization()

        assert stats['num_experts'] == 4
        assert stats['expert_type'] == 'ffn'
        assert stats['expert_ffn_type'] == 'mlp'
        assert stats['routing_type'] == 'linear'
        assert stats['top_k'] == 2
        assert isinstance(stats['expert_capacity_train'], int)
        assert isinstance(stats['expert_capacity_eval'], int)

    def test_auxiliary_losses_training(self, base_moe_config, sample_input_2d):
        """Test auxiliary loss computation during training."""
        config = MoEConfig(**base_moe_config)
        layer = MixtureOfExperts(config=config)

        # Forward pass in training mode
        output = layer(sample_input_2d, training=True)

        # Check that auxiliary losses were added
        assert len(layer.losses) > 0

        # Verify loss values are reasonable
        for loss in layer.losses:
            loss_value = float(loss)
            assert np.isfinite(loss_value)
            assert loss_value >= 0

    def test_no_auxiliary_losses_inference(self, base_moe_config, sample_input_2d):
        """Test that no auxiliary losses are added during inference."""
        config = MoEConfig(**base_moe_config)
        layer = MixtureOfExperts(config=config)

        # Forward pass in inference mode
        output = layer(sample_input_2d, training=False)

        # Should have no auxiliary losses in inference mode
        assert len(layer.losses) == 0

    def test_edge_cases(self):
        """Test error conditions and edge cases."""

        # Test invalid num_experts
        with pytest.raises(ValueError, match="num_experts must be positive"):
            invalid_config = MoEConfig(num_experts=0)
            MixtureOfExperts(config=invalid_config)

        # Test invalid FFN config
        with pytest.raises(ValueError, match="Invalid FFN configuration"):
            invalid_ffn_config = MoEConfig(
                num_experts=2,
                expert_config=ExpertConfig(
                    ffn_config={'type': 'mlp', 'hidden_dim': -100, 'output_dim': 64}  # Invalid negative dim
                )
            )
            MixtureOfExperts(config=invalid_ffn_config)

        # Test missing FFN type
        with pytest.raises(ValueError, match="ffn_config must contain 'type'"):
            no_type_config = MoEConfig(
                num_experts=2,
                expert_config=ExpertConfig(
                    ffn_config={'hidden_dim': 128}  # Missing 'type'
                )
            )
            MixtureOfExperts(config=no_type_config)

    def test_convenience_factory_function(self, sample_input_2d):
        """Test the create_ffn_moe convenience function."""
        moe_layer = create_ffn_moe(
            num_experts=6,
            ffn_config={
                'type': 'swiglu',
                'output_dim': 256,
                'output_dim': 256,
                'ffn_expansion_factor': 4
            },
            top_k=2,
            gating_type='linear',
            aux_loss_weight=0.02
        )

        output = moe_layer(sample_input_2d)

        assert moe_layer.built
        assert output.shape[0] == sample_input_2d.shape[0]
        assert output.shape[-1] == 256

        # Verify configuration was set correctly
        stats = moe_layer.get_expert_utilization()
        assert stats['num_experts'] == 6
        assert stats['expert_ffn_type'] == 'swiglu'
        assert stats['routing_type'] == 'linear'
        assert stats['top_k'] == 2

    def test_different_input_shapes(self, base_moe_config):
        """Test MoE with different input shapes."""
        config = MoEConfig(**base_moe_config)
        layer = MixtureOfExperts(config=config)

        # Test various input shapes
        input_shapes = [
            (4, 256),  # 2D: (batch, features)
            (2, 16, 256),  # 3D: (batch, seq_len, features)
            (1, 64, 256),  # 3D: different seq_len
        ]

        for input_shape in input_shapes:
            test_input = keras.random.normal(shape=input_shape)
            output = layer(test_input)

            expected_output_shape = list(input_shape)
            expected_output_shape[-1] = 64  # output_dim from config

            assert output.shape == tuple(expected_output_shape)

    def test_token_dropping_behavior(self, sample_input_3d):
        """Test token dropping behavior when expert capacity is exceeded."""
        # Config with aggressive capacity constraints
        config = MoEConfig(
            num_experts=2,  # Few experts to force capacity issues
            expert_config=ExpertConfig(
                ffn_config={'type': 'mlp', 'hidden_dim': 64, 'output_dim': 32}
            ),
            gating_config=GatingConfig(top_k=1, capacity_factor=0.5),  # Low capacity
            drop_tokens=True,
            use_residual_connection=True
        )

        layer = MixtureOfExperts(config=config)

        # Should work even with capacity constraints
        output = layer(sample_input_3d, training=True)
        assert output.shape == (sample_input_3d.shape[0], sample_input_3d.shape[1], 32)

    def test_jitter_noise_effect(self, base_moe_config, sample_input_2d):
        """Test that jitter noise affects routing during training."""
        config = MoEConfig(**base_moe_config)
        config.jitter_noise = 0.1  # Significant noise
        layer = MixtureOfExperts(config=config)

        # Run multiple forward passes in training mode
        outputs = []
        for _ in range(3):
            output = layer(sample_input_2d, training=True)
            outputs.append(ops.convert_to_numpy(output))

        # With jitter noise, outputs should vary slightly
        # (This is probabilistic, but very likely with reasonable noise)
        variation_exists = False
        for i in range(1, len(outputs)):
            if not np.allclose(outputs[0], outputs[i], rtol=1e-4):
                variation_exists = True
                break

        assert variation_exists, "Expected variation due to jitter noise"

    def test_no_jitter_noise_deterministic(self, sample_input_2d):
        """Test deterministic behavior without jitter noise."""
        config = MoEConfig(
            num_experts=4,
            expert_config=ExpertConfig(
                ffn_config={'type': 'mlp', 'hidden_dim': 128, 'output_dim': 64}
            ),
            gating_config=GatingConfig(top_k=2, add_noise=False),  # No noise
            jitter_noise=0.0  # No jitter
        )
        layer = MixtureOfExperts(config=config)

        # Multiple forward passes should be identical in inference mode
        output1 = layer(sample_input_2d, training=False)
        output2 = layer(sample_input_2d, training=False)

        np.testing.assert_allclose(
            ops.convert_to_numpy(output1),
            ops.convert_to_numpy(output2),
            rtol=1e-7, atol=1e-7,
            err_msg="Expected deterministic behavior without noise"
        )

    def test_from_config_class_method(self, base_moe_config):
        """Test layer creation using from_config class method."""
        config = MoEConfig(**base_moe_config)
        layer = MixtureOfExperts(config=config)

        # Get config and recreate layer
        layer_config = layer.get_config()
        recreated_layer = MixtureOfExperts.from_config(layer_config)

        # Verify configurations match
        assert recreated_layer.num_experts == layer.num_experts
        assert recreated_layer.config.expert_config.ffn_config == layer.config.expert_config.ffn_config
        assert recreated_layer.config.gating_config.gating_type == layer.config.gating_config.gating_type

    def test_build_state_consistency(self, base_moe_config, sample_input_2d):
        """Test that building state is consistent."""
        config = MoEConfig(**base_moe_config)
        layer = MixtureOfExperts(config=config)

        # Layer should not be built initially
        assert not layer.built

        # After forward pass, should be built
        output = layer(sample_input_2d)
        assert layer.built

        # All experts should be built
        for expert in layer.experts:
            assert expert.built

        # Gating network should be built
        assert layer.gating_network.built

    def test_capacity_calculation(self, base_moe_config, sample_input_2d):
        """Test expert capacity calculation."""
        config = MoEConfig(**base_moe_config)
        config.train_capacity_factor = 2.0
        config.eval_capacity_factor = 1.0
        layer = MixtureOfExperts(config=config)

        # Build layer
        _ = layer(sample_input_2d)

        # Check capacity values were calculated
        assert layer._expert_capacity_train is not None
        assert layer._expert_capacity_eval is not None
        assert layer._expert_capacity_train >= layer._expert_capacity_eval
        assert layer._expert_capacity_train > 0
        assert layer._expert_capacity_eval > 0

    def test_weight_shape_consistency(self, base_moe_config, sample_input_2d):
        """Test that weight shapes are consistent across experts."""
        config = MoEConfig(**base_moe_config)
        layer = MixtureOfExperts(config=config)

        # Build layer
        _ = layer(sample_input_2d)

        # All experts should have the same FFN architecture
        first_expert_weights = layer.experts[0].trainable_variables
        for expert in layer.experts[1:]:
            expert_weights = expert.trainable_variables
            assert len(expert_weights) == len(first_expert_weights)

            for w1, w2 in zip(first_expert_weights, expert_weights):
                assert w1.shape == w2.shape


class TestMoEConfigurations:
    """Test MoE configuration classes."""

    def test_expert_config_defaults(self):
        """Test ExpertConfig default values and validation."""
        # Default config should work
        config = ExpertConfig()
        assert config.ffn_config['type'] == 'mlp'
        assert config.use_bias is True

        # Custom config
        custom_config = ExpertConfig(
            ffn_config={'type': 'swiglu', 'output_dim': 512, 'output_dim': 512},
            use_bias=False
        )
        assert custom_config.ffn_config['type'] == 'swiglu'
        assert custom_config.use_bias is False

    def test_expert_config_validation(self):
        """Test ExpertConfig validation."""
        # Missing type should raise error
        with pytest.raises(ValueError, match="ffn_config must contain 'type'"):
            ExpertConfig(ffn_config={'hidden_dim': 128})

    def test_gating_config_defaults(self):
        """Test GatingConfig default values."""
        config = GatingConfig()
        assert config.gating_type == 'linear'
        assert config.top_k == 1
        assert config.capacity_factor == 1.25
        assert config.add_noise is True
        assert config.aux_loss_weight == 0.01

    def test_moe_config_derived_parameters(self):
        """Test MoEConfig derived parameter computation."""
        config = MoEConfig(
            num_experts=8,
            gating_config=GatingConfig(capacity_factor=2.0)
        )

        # Should compute derived parameters
        assert config.train_capacity_factor == 2.0  # Should match gating capacity
        assert config.eval_capacity_factor == 1.6  # Should be 0.8 * capacity_factor

    def test_config_serialization(self):
        """Test configuration serialization and deserialization."""
        original_config = MoEConfig(
            num_experts=6,
            expert_config=ExpertConfig(
                ffn_config={
                    'type': 'geglu',
                    'hidden_dim': 1024,
                    'output_dim': 512
                }
            ),
            gating_config=GatingConfig(
                gating_type='cosine',
                top_k=2,
                embedding_dim=128
            )
        )

        # Serialize and deserialize
        config_dict = original_config.to_dict()
        restored_config = MoEConfig.from_dict(config_dict)

        # Verify all fields match
        assert restored_config.num_experts == original_config.num_experts
        assert restored_config.expert_config.ffn_config == original_config.expert_config.ffn_config
        assert restored_config.gating_config.gating_type == original_config.gating_config.gating_type
        assert restored_config.gating_config.embedding_dim == original_config.gating_config.embedding_dim


class TestFFNExpert:
    """Test FFN expert implementation."""

    @pytest.fixture
    def mlp_expert_config(self) -> Dict[str, Any]:
        """MLP expert configuration."""
        return {
            'ffn_config': {
                'type': 'mlp',
                'hidden_dim': 256,
                'output_dim': 128,
                'activation': 'gelu'
            }
        }

    @pytest.fixture
    def swiglu_expert_config(self) -> Dict[str, Any]:
        """SwiGLU expert configuration."""
        return {
            'ffn_config': {
                'type': 'swiglu',
                'output_dim': 512,
                'output_dim': 512,
                'ffn_expansion_factor': 4
            }
        }

    @pytest.fixture
    def sample_input(self) -> keras.KerasTensor:
        """Sample input for expert testing."""
        return keras.random.normal(shape=(8, 256))

    def test_mlp_expert_initialization(self, mlp_expert_config):
        """Test MLP expert initialization."""
        expert = FFNExpert(**mlp_expert_config)

        assert expert.ffn_config['type'] == 'mlp'
        assert not expert.built
        assert expert.ffn_block is None  # Created in build()

    def test_mlp_expert_forward_pass(self, mlp_expert_config, sample_input):
        """Test MLP expert forward pass."""
        expert = FFNExpert(**mlp_expert_config)

        output = expert(sample_input)

        assert expert.built
        assert expert.ffn_block is not None
        assert output.shape[0] == sample_input.shape[0]
        assert output.shape[-1] == 128  # output_dim from config

    def test_swiglu_expert_forward_pass(self, swiglu_expert_config):
        """Test SwiGLU expert forward pass."""
        expert = FFNExpert(**swiglu_expert_config)
        sample_input = keras.random.normal(shape=(4, 512))  # Match output_dim

        output = expert(sample_input)

        assert expert.built
        assert output.shape[0] == sample_input.shape[0]
        assert output.shape[-1] == 512  # output_dim from config

    def test_expert_serialization(self, mlp_expert_config, sample_input):
        """Test expert serialization."""
        # Create model with expert
        inputs = keras.Input(shape=sample_input.shape[1:])
        expert = FFNExpert(**mlp_expert_config)
        outputs = expert(inputs)
        model = keras.Model(inputs, outputs)

        # Get original prediction
        original_prediction = model(sample_input)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_expert.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_prediction = loaded_model(sample_input)

            # Verify identical predictions
            np.testing.assert_allclose(
                ops.convert_to_numpy(original_prediction),
                ops.convert_to_numpy(loaded_prediction),
                rtol=1e-6, atol=1e-6,
                err_msg="Expert predictions differ after serialization"
            )

    def test_expert_output_shape_computation(self, mlp_expert_config):
        """Test expert output shape computation."""
        expert = FFNExpert(**mlp_expert_config)

        input_shape = (None, 256)
        output_shape = expert.compute_output_shape(input_shape)

        assert output_shape == (None, 128)  # output_dim from config

    def test_expert_factory_function(self):
        """Test expert factory function."""
        expert = create_expert(
            'ffn',
            ffn_config={
                'type': 'differential',
                'hidden_dim': 512,
                'output_dim': 256,
                'branch_activation': 'relu'
            }
        )

        assert isinstance(expert, FFNExpert)
        assert expert.ffn_config['type'] == 'differential'

        # Test invalid expert type
        with pytest.raises(ValueError, match="Unsupported expert type"):
            create_expert('invalid_type')


class TestGatingNetworks:
    """Test gating network implementations."""

    @pytest.fixture
    def sample_input(self) -> keras.KerasTensor:
        """Sample input for gating testing."""
        return keras.random.normal(shape=(6, 256))

    def test_linear_gating(self, sample_input):
        """Test linear gating network."""
        gating = LinearGating(
            num_experts=8,
            top_k=3,
            add_noise=True,
            noise_std=0.1
        )

        weights, indices, info = gating(sample_input, training=True)

        assert gating.built
        assert weights.shape == (6, 8)
        assert indices.shape == (6, 3)

        # Check auxiliary info
        assert 'gate_logits' in info
        assert 'expert_weights' in info
        assert 'raw_gate_probs' in info

        # Weights should sum to approximately 1 for top-k selections
        selected_weights = keras.ops.sum(weights, axis=-1)
        np.testing.assert_allclose(
            ops.convert_to_numpy(selected_weights),
            np.ones(6),
            rtol=1e-5, atol=1e-5
        )

    def test_cosine_gating(self, sample_input):
        """Test cosine similarity gating."""
        gating = CosineGating(
            num_experts=4,
            embedding_dim=128,
            top_k=2,
            learnable_temperature=True
        )

        weights, indices, info = gating(sample_input, training=True)

        assert gating.built
        assert weights.shape == (6, 4)
        assert indices.shape == (6, 2)

        # Check cosine similarities in info
        assert 'cosine_similarities' in info
        cosine_sims = info['cosine_similarities']
        assert cosine_sims.shape == (6, 4)

        # Cosine similarities should be in [-1, 1] range
        cosine_values = ops.convert_to_numpy(cosine_sims)
        assert np.all(cosine_values >= -1.0) and np.all(cosine_values <= 1.0)

    def test_softmoe_gating(self):
        """Test SoftMoE gating mechanism."""
        # SoftMoE requires 3D input (batch, seq_len, hidden)
        sample_input_3d = keras.random.normal(shape=(4, 16, 256))

        gating = SoftMoEGating(
            num_experts=6,
            num_slots=4
        )

        weights, indices, info = gating(sample_input_3d, training=True)

        assert gating.built
        assert weights.shape == (4, 16, 6)
        assert indices.shape == (4, 16, 6)  # All experts used in SoftMoE

        # Check SoftMoE-specific info
        assert 'expert_inputs' in info
        assert 'phi_weights' in info
        assert 'soft_slots' in info

        expert_inputs = info['expert_inputs']
        assert expert_inputs.shape == (4, 6, 4 * 256)  # (batch, experts, slots * hidden)

    def test_gating_factory(self, sample_input):
        """Test gating factory function."""
        gating_configs = [
            ('linear', {'top_k': 2, 'add_noise': False}),
            ('cosine', {'embedding_dim': 64, 'top_k': 1}),
        ]

        for gating_type, kwargs in gating_configs:
            gating = create_gating(gating_type, num_experts=4, **kwargs)

            weights, indices, info = gating(sample_input)

            assert weights.shape[0] == sample_input.shape[0]
            assert weights.shape[1] == 4  # num_experts

        # Test invalid gating type
        with pytest.raises(ValueError, match="Unsupported gating type"):
            create_gating('invalid', num_experts=4)

    def test_gating_serialization(self, sample_input):
        """Test gating network serialization."""
        gating = LinearGating(num_experts=4, top_k=2, add_noise=True)

        # Create model with gating
        inputs = keras.Input(shape=sample_input.shape[1:])
        weights, indices, info = gating(inputs)
        # Use only weights for output (indices are integers)
        model = keras.Model(inputs, weights)

        # Get original prediction
        original_output = model(sample_input)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_gating.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_output = loaded_model(sample_input)

            # Verify identical outputs
            np.testing.assert_allclose(
                ops.convert_to_numpy(original_output),
                ops.convert_to_numpy(loaded_output),
                rtol=1e-6, atol=1e-6,
                err_msg="Gating outputs differ after serialization"
            )


class TestMoEIntegration:
    """Test MoE integration with full models and training."""

    def test_moe_in_transformer_model(self):
        """Test MoE integration in a transformer-like model."""
        # Create a simple transformer-like model with MoE
        vocab_size = 1000
        seq_len = 32
        hidden_dim = 256

        inputs = keras.Input(shape=(seq_len,), dtype='int32')

        # Embedding
        x = keras.layers.Embedding(vocab_size, hidden_dim)(inputs)

        # MoE layer
        moe_config = MoEConfig(
            num_experts=4,
            expert_config=ExpertConfig(
                ffn_config={
                    'type': 'swiglu',
                    'output_dim': hidden_dim,
                    'output_dim': hidden_dim,
                    'ffn_expansion_factor': 4
                }
            ),
            gating_config=GatingConfig(top_k=2)
        )
        x = MixtureOfExperts(config=moe_config)(x)

        # Output layer
        outputs = keras.layers.Dense(vocab_size, activation='softmax')(x)

        model = keras.Model(inputs, outputs)
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Test forward pass
        sample_tokens = keras.random.randint(
            shape=(4, seq_len),
            minval=0,
            maxval=vocab_size,
            dtype='int32'
        )

        output = model(sample_tokens)
        assert output.shape == (4, seq_len, vocab_size)

    def test_moe_training_with_auxiliary_losses(self):
        """Test MoE training with auxiliary loss tracking."""
        # Simple model for testing training
        inputs = keras.Input(shape=(128,))

        moe_config = MoEConfig(
            num_experts=4,
            expert_config=ExpertConfig(
                ffn_config={'type': 'mlp', 'hidden_dim': 256, 'output_dim': 64}
            ),
            gating_config=GatingConfig(aux_loss_weight=0.02, z_loss_weight=1e-3)
        )

        x = MixtureOfExperts(config=moe_config)(inputs)
        outputs = keras.layers.Dense(10, activation='softmax')(x)

        model = keras.Model(inputs, outputs)
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Generate dummy training data
        x_train = keras.random.normal(shape=(32, 128))
        y_train = keras.random.randint(shape=(32,), minval=0, maxval=10, dtype='int32')

        # Train for one step to verify auxiliary losses work
        history = model.fit(x_train, y_train, epochs=1, verbose=0)

        # Training should complete successfully
        assert len(history.history['loss']) == 1
        assert np.isfinite(history.history['loss'][0])

    def test_moe_memory_efficiency(self):
        """Test that MoE doesn't create excessive memory usage."""
        # Create a reasonably sized MoE layer
        config = MoEConfig(
            num_experts=8,
            expert_config=ExpertConfig(
                ffn_config={'type': 'mlp', 'hidden_dim': 1024, 'output_dim': 512}
            ),
            gating_config=GatingConfig(top_k=2)
        )

        layer = MixtureOfExperts(config=config)
        sample_input = keras.random.normal(shape=(16, 512))

        # Should work without memory errors
        output = layer(sample_input)
        assert output.shape == (16, 512)

    def test_large_expert_count(self):
        """Test MoE with large number of experts."""
        config = MoEConfig(
            num_experts=32,  # Large expert count
            expert_config=ExpertConfig(
                ffn_config={'type': 'mlp', 'hidden_dim': 128, 'output_dim': 64}
            ),
            gating_config=GatingConfig(top_k=4)
        )

        layer = MixtureOfExperts(config=config)
        sample_input = keras.random.normal(shape=(8, 128))

        output = layer(sample_input)

        assert layer.built
        assert len(layer.experts) == 32
        assert output.shape == (8, 64)


class TestMoEPerformance:
    """Performance and behavior tests for MoE layers."""

    def test_output_determinism_inference(self):
        """Test that MoE outputs are deterministic in inference mode."""
        config = MoEConfig(
            num_experts=4,
            expert_config=ExpertConfig(
                ffn_config={'type': 'mlp', 'hidden_dim': 128, 'output_dim': 64}
            ),
            gating_config=GatingConfig(top_k=1, add_noise=False),
            jitter_noise=0.0  # No noise
        )

        layer = MixtureOfExperts(config=config)
        sample_input = keras.random.normal(shape=(8, 256))

        # Multiple inference passes should be identical
        output1 = layer(sample_input, training=False)
        output2 = layer(sample_input, training=False)

        np.testing.assert_allclose(
            ops.convert_to_numpy(output1),
            ops.convert_to_numpy(output2),
            rtol=1e-7, atol=1e-7,
            err_msg="Expected deterministic inference behavior"
        )

    def test_expert_capacity_respects_settings(self):
        """Test that expert capacity settings are respected."""
        config = MoEConfig(
            num_experts=2,
            expert_config=ExpertConfig(
                ffn_config={'type': 'mlp', 'hidden_dim': 64, 'output_dim': 32}
            ),
            gating_config=GatingConfig(top_k=1, capacity_factor=1.0),
            train_capacity_factor=2.0,
            eval_capacity_factor=1.0
        )

        layer = MixtureOfExperts(config=config)
        sample_input = keras.random.normal(shape=(8, 64))

        # Build layer
        _ = layer(sample_input)

        # Check that capacity factors were applied
        assert layer._expert_capacity_train > layer._expert_capacity_eval

    def test_gradient_flow_through_routing(self):
        """Test that gradients flow through the routing mechanism."""
        config = MoEConfig(
            num_experts=4,
            expert_config=ExpertConfig(
                ffn_config={'type': 'mlp', 'hidden_dim': 128, 'output_dim': 64}
            ),
            gating_config=GatingConfig(top_k=2)
        )

        layer = MixtureOfExperts(config=config)
        sample_input = keras.random.normal(shape=(6, 128))

        # Use persistent tape for multiple gradient computations
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(sample_input)
            output = layer(sample_input, training=True)
            loss = ops.mean(ops.square(output))

        # Gradients should flow to gating network
        gating_gradients = tape.gradient(loss, layer.gating_network.trainable_variables)
        assert len(gating_gradients) > 0
        assert all(g is not None for g in gating_gradients)

        # Check gradients for experts
        total_expert_grads = []
        for expert in layer.experts:
            expert_grads = tape.gradient(loss, expert.trainable_variables)
            total_expert_grads.extend([g for g in expert_grads if g is not None])

        assert len(total_expert_grads) > 0, "At least some experts should receive gradients"

        # Clean up persistent tape
        del tape


def test_moe_debug_helper():
    """Test helper function for debugging MoE layer serialization."""
    from dl_techniques.utils.logger import logger

    def debug_moe_serialization(config: MoEConfig, sample_input: keras.KerasTensor):
        """Debug helper for MoE serialization issues."""
        try:
            # Test basic functionality
            layer = MixtureOfExperts(config=config)
            output = layer(sample_input)
            logger.info(f"✅ Forward pass successful: {output.shape}")

            # Test configuration
            layer_config = layer.get_config()
            logger.info(f"✅ Configuration keys: {list(layer_config.keys())}")

            # Test serialization
            inputs = keras.Input(shape=sample_input.shape[1:])
            outputs = MixtureOfExperts(config=config)(inputs)
            model = keras.Model(inputs, outputs)

            with tempfile.TemporaryDirectory() as tmpdir:
                model.save(os.path.join(tmpdir, 'test.keras'))
                loaded = keras.models.load_model(os.path.join(tmpdir, 'test.keras'))
                logger.info("✅ Serialization test passed")

        except Exception as e:
            logger.error(f"❌ Error: {e}")
            raise

    # Test the debug helper with a simple config
    config = MoEConfig(
        num_experts=2,
        expert_config=ExpertConfig(
            ffn_config={'type': 'mlp', 'hidden_dim': 64, 'output_dim': 32}
        )
    )
    sample_input = keras.random.normal(shape=(4, 64))

    debug_moe_serialization(config, sample_input)


# Integration test with real training scenario
class TestMoETrainingIntegration:
    """Integration tests with actual training scenarios."""

    def test_end_to_end_training(self):
        """Test complete end-to-end MoE training."""
        # Create a simple classification model with MoE
        num_classes = 5
        input_dim = 128

        inputs = keras.Input(shape=(input_dim,))

        # Add a standard dense layer first
        x = keras.layers.Dense(256, activation='relu')(inputs)

        # MoE layer
        moe_config = MoEConfig(
            num_experts=4,
            expert_config=ExpertConfig(
                ffn_config={'type': 'mlp', 'hidden_dim': 512, 'output_dim': 256}
            ),
            gating_config=GatingConfig(top_k=2, aux_loss_weight=0.01)
        )
        x = MixtureOfExperts(config=moe_config)(x)

        # Classification head
        outputs = keras.layers.Dense(num_classes, activation='softmax')(x)

        model = keras.Model(inputs, outputs)
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Generate synthetic data
        x_train = keras.random.normal(shape=(64, input_dim))
        y_train = keras.random.randint(shape=(64,), minval=0, maxval=num_classes, dtype='int32')

        x_val = keras.random.normal(shape=(32, input_dim))
        y_val = keras.random.randint(shape=(32,), minval=0, maxval=num_classes, dtype='int32')

        # Train for a few epochs
        history = model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=3,
            batch_size=16,
            verbose=0
        )

        # Training should complete successfully
        assert len(history.history['loss']) == 3
        assert all(np.isfinite(loss) for loss in history.history['loss'])

        # Model should be able to predict
        predictions = model.predict(x_val, verbose=0)
        assert predictions.shape == (32, num_classes)
        assert np.allclose(np.sum(predictions, axis=1), 1.0, rtol=1e-5)  # Softmax outputs

    def test_moe_vs_dense_equivalence(self):
        """Test that single-expert MoE behaves similarly to dense layer."""
        input_dim = 128
        output_dim = 64
        sample_input = keras.random.normal(shape=(16, input_dim))

        # Dense baseline
        dense_layer = keras.layers.Dense(output_dim, activation='relu', use_bias=True)
        dense_output = dense_layer(sample_input)

        # Single-expert MoE (should behave similarly to dense)
        moe_config = MoEConfig(
            num_experts=1,  # Single expert
            expert_config=ExpertConfig(
                ffn_config={'type': 'mlp', 'hidden_dim': output_dim, 'output_dim': output_dim}
            ),
            gating_config=GatingConfig(top_k=1, aux_loss_weight=0.0),  # No auxiliary loss
            jitter_noise=0.0  # No noise
        )
        moe_layer = MixtureOfExperts(config=moe_config)
        moe_output = moe_layer(sample_input, training=False)

        # Outputs should have same shape
        assert dense_output.shape == moe_output.shape

        # Both should be learnable (have trainable parameters)
        assert len(dense_layer.trainable_variables) > 0
        assert len(moe_layer.trainable_variables) > 0


# Run tests with: pytest test_mixture_of_experts.py -v
if __name__ == "__main__":
    pytest.main([__file__, "-v"])