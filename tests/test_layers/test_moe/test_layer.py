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
from dl_techniques.layers.moe.gating import (
    LinearGating, CosineGating, SoftMoEGating, create_gating, compute_auxiliary_loss
)


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
        assert stats['drop_tokens'] is True
        assert stats['use_residual_connection'] is True

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

        # Test invalid num_experts. Since step 4 (B1) this is rejected by
        # ``MoEConfig.__post_init__`` itself, before ``MixtureOfExperts`` ever
        # sees it -- the config object can no longer be constructed at all.
        with pytest.raises(ValueError, match="num_experts must be >= 1"):
            MoEConfig(num_experts=0)

        # ``MixtureOfExperts``'s own guard is retained as defence in depth and
        # still fires when the (mutable) dataclass is mutated after construction.
        mutated = MoEConfig(num_experts=2)
        mutated.num_experts = 0
        with pytest.raises(ValueError, match="num_experts must be positive"):
            MixtureOfExperts(config=mutated)

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
            gating_config=GatingConfig(top_k=1),
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
        assert config.add_noise is True
        assert config.aux_loss_weight == 0.01

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
        assert expert.ffn_block is not None  # Created in __init__ (Golden Rule)

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
        assert 'dispatch_weights' in info
        assert 'combine_weights' in info
        assert 'soft_slots' in info

        expert_inputs = info['expert_inputs']
        assert expert_inputs.shape == (4, 6, 4 * 256)  # (batch, experts, slots * hidden)

        # Dispatch weights sum to 1 over the sequence axis for each (expert, slot).
        dispatch_sum = ops.convert_to_numpy(ops.sum(info['dispatch_weights'], axis=1))
        np.testing.assert_allclose(dispatch_sum, np.ones_like(dispatch_sum), atol=1e-5)

        # Combine weights sum to 1 over (experts * slots) for each token.
        combine_sum = ops.convert_to_numpy(ops.sum(info['combine_weights'], axis=(-2, -1)))
        np.testing.assert_allclose(combine_sum, np.ones_like(combine_sum), atol=1e-5)

        # Marginalized expert_weights are now non-uniform (no longer 1/N).
        ew = ops.convert_to_numpy(weights)
        assert ew.std() > 1e-4, "Expected non-uniform per-expert weights after A2 fix"

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


class TestReviewFixes:
    """Tests covering the May-2026 MoE review fixes (A1-A5, B2, B4)."""

    def test_cosine_temperature_divides_softmax(self):
        """Larger ``temperature`` should produce a flatter softmax (A3)."""
        rng = np.random.default_rng(0)
        x = ops.convert_to_tensor(rng.standard_normal((32, 64)).astype(np.float32))

        # Same kernel init seeds so the only varying factor is temperature.
        keras.utils.set_random_seed(123)
        low_t = CosineGating(num_experts=8, embedding_dim=32, top_k=8,
                              temperature=0.5, learnable_temperature=False)
        keras.utils.set_random_seed(123)
        high_t = CosineGating(num_experts=8, embedding_dim=32, top_k=8,
                               temperature=4.0, learnable_temperature=False)

        w_low, _, _ = low_t(x, training=False)
        w_high, _, _ = high_t(x, training=False)

        # Entropy of weights: higher temperature -> flatter -> larger entropy.
        def entropy(p):
            p = ops.convert_to_numpy(p)
            p = np.clip(p, 1e-12, 1.0)
            return float((-p * np.log(p)).sum(axis=-1).mean())

        assert entropy(w_high) > entropy(w_low) + 1e-3, (
            f"Expected higher temperature -> larger entropy "
            f"(low_t={entropy(w_low):.4f}, high_t={entropy(w_high):.4f})"
        )

    def test_softmoe_dispatch_and_combine_separate(self):
        """Dispatch and combine softmaxes operate over different axes (A1)."""
        x = keras.random.normal(shape=(2, 12, 32))
        gating = SoftMoEGating(num_experts=4, num_slots=3)
        _, _, info = gating(x, training=False)

        dispatch = ops.convert_to_numpy(info['dispatch_weights'])  # [b, s, e, l]
        combine = ops.convert_to_numpy(info['combine_weights'])    # [b, s, e, l]

        # Dispatch sums to 1 over seq axis (axis=1) for each (expert, slot).
        np.testing.assert_allclose(dispatch.sum(axis=1),
                                    np.ones((2, 4, 3)), atol=1e-5)
        # Combine sums to 1 over (experts * slots) for each token.
        np.testing.assert_allclose(combine.sum(axis=(-2, -1)),
                                    np.ones((2, 12)), atol=1e-5)
        # Distinct tensors: with random init, the two softmaxes shouldn't match.
        assert not np.allclose(dispatch, combine, atol=1e-4)

    def test_capacity_factor_fields_removed_from_config(self):
        """B4: train/eval_capacity_factor are gone; from_dict ignores them."""
        cfg = MoEConfig(num_experts=2)
        assert not hasattr(cfg, 'train_capacity_factor')
        assert not hasattr(cfg, 'eval_capacity_factor')

        # Round-trip with legacy keys still works (they're silently dropped).
        legacy = cfg.to_dict()
        legacy['train_capacity_factor'] = 2.0
        legacy['eval_capacity_factor'] = 1.0
        restored = MoEConfig.from_dict(legacy)
        assert restored.num_experts == 2

    # --- C1 (F-6): routing_dtype / capacity_factor removed ----------------

    def test_dead_fields_removed_from_config_surface(self):
        """C1: `routing_dtype` and `capacity_factor` are gone from the dataclasses.

        Both were accepted, validated and serialized while gating no behaviour;
        `routing_dtype` additionally accepted any string. Neither may be
        constructible or serialized any more.
        """
        moe = MoEConfig(num_experts=2)
        gating = GatingConfig()
        assert not hasattr(moe, 'routing_dtype')
        assert not hasattr(gating, 'capacity_factor')
        assert 'routing_dtype' not in moe.to_dict()
        assert 'capacity_factor' not in moe.to_dict()['gating_config']

        with pytest.raises(TypeError):
            MoEConfig(num_experts=2, routing_dtype='float32')
        with pytest.raises(TypeError):
            GatingConfig(capacity_factor=1.25)

    def test_diagnostic_flags_gate_no_forward_behaviour(self):
        """C1/5c: `drop_tokens`/`use_residual_connection` are diagnostic-only.

        Pins the claim the rewritten docstrings now make. Flipping both must
        leave the forward output bit-identical while still being echoed by
        ``get_expert_utilization()``.
        """
        def build(flag):
            keras.utils.set_random_seed(11)
            return MixtureOfExperts(MoEConfig(
                num_experts=4,
                expert_config=ExpertConfig(
                    ffn_config={'type': 'mlp', 'hidden_dim': 16, 'output_dim': 10}
                ),
                gating_config=GatingConfig(top_k=2, add_noise=False),
                jitter_noise=0.0,
                drop_tokens=flag,
                use_residual_connection=flag,
            ))

        x = ops.convert_to_tensor(
            np.arange(2 * 5 * 10, dtype='float32').reshape(2, 5, 10) / 100.0)
        on, off = build(True), build(False)
        y_on = ops.convert_to_numpy(on(x, training=False))
        y_off = ops.convert_to_numpy(off(x, training=False))

        np.testing.assert_array_equal(y_on, y_off)
        assert on.get_expert_utilization()['drop_tokens'] is True
        assert off.get_expert_utilization()['use_residual_connection'] is False

    def test_gating_pre_norm_via_factory(self):
        """B2: GatingConfig.norm_type wires pre-gating norm via the factory."""
        cfg = MoEConfig(
            num_experts=2,
            expert_config=ExpertConfig(
                ffn_config={'type': 'mlp', 'hidden_dim': 32, 'output_dim': 16}
            ),
            gating_config=GatingConfig(
                top_k=1, gating_type='linear',
                norm_type='rms_norm', norm_config={'epsilon': 1e-5},
            ),
        )
        layer = MixtureOfExperts(config=cfg)
        x = keras.random.normal(shape=(4, 16))
        y = layer(x, training=False)
        assert y.shape == (4, 16)
        assert layer.gating_network.pre_norm is not None

    def test_expert_pre_norm_via_factory(self):
        """B2: ExpertConfig.norm_type wires per-expert pre-norm via the factory."""
        cfg = MoEConfig(
            num_experts=2,
            expert_config=ExpertConfig(
                ffn_config={'type': 'mlp', 'hidden_dim': 32, 'output_dim': 16},
                norm_type='layer_norm',
                pre_norm=True,
                post_norm=False,
            ),
            gating_config=GatingConfig(top_k=1),
        )
        layer = MixtureOfExperts(config=cfg)
        x = keras.random.normal(shape=(4, 16))
        _ = layer(x, training=False)
        for expert in layer.experts:
            assert expert.pre_norm is not None
            assert expert.post_norm is None

    def test_norm_factory_serialization_roundtrip(self):
        """Save/load round-trip with pre-gating + per-expert norms (B2)."""
        cfg = MoEConfig(
            num_experts=2,
            expert_config=ExpertConfig(
                ffn_config={'type': 'mlp', 'hidden_dim': 32, 'output_dim': 16},
                norm_type='rms_norm',
            ),
            gating_config=GatingConfig(top_k=1, norm_type='layer_norm'),
        )
        inputs = keras.Input(shape=(16,))
        outputs = MixtureOfExperts(config=cfg)(inputs)
        model = keras.Model(inputs, outputs)

        x = keras.random.normal(shape=(4, 16))
        y_ref = model(x, training=False)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'moe_norms.keras')
            model.save(path)
            restored = keras.models.load_model(path)

        y_new = restored(x, training=False)
        np.testing.assert_allclose(
            ops.convert_to_numpy(y_ref),
            ops.convert_to_numpy(y_new),
            atol=1e-5,
        )


class TestMoEReviewRegressions:
    """Regressions pinning the iter-1 MoE review fixes.

    Covers: build() idempotency guard, SoftMoE aux-info contract shape,
    GatingConfig validation symmetry, and CosineGating temperature robustness.
    """

    # --- F1: build() idempotency (if self.built: return) -----------------

    def _moe(self, gating_type: str, top_k: int = 2, num_experts: int = 4):
        ec = ExpertConfig(ffn_config={'type': 'mlp', 'hidden_dim': 32, 'output_dim': 16})
        gc = GatingConfig(gating_type=gating_type, top_k=top_k)
        return MixtureOfExperts(MoEConfig(num_experts=num_experts, expert_config=ec, gating_config=gc))

    @pytest.mark.parametrize("gating_type,top_k", [
        ('linear', 2), ('cosine', 2), ('softmoe', 1),
    ])
    def test_moe_double_build_idempotent(self, gating_type, top_k):
        """Calling build() twice must be a no-op, not raise."""
        layer = self._moe(gating_type, top_k)
        layer.build((2, 5, 16))
        layer.build((2, 5, 16))  # must not raise
        assert layer.built

    def test_gating_layers_double_build_idempotent(self):
        """All three gating layers and FFNExpert tolerate a second build()."""
        gatings = [
            LinearGating(num_experts=4, top_k=2),
            CosineGating(num_experts=4, top_k=2, embedding_dim=8),
            SoftMoEGating(num_experts=4, num_slots=3),
        ]
        for g in gatings:
            g.build((2, 5, 16))
            g.build((2, 5, 16))  # must not raise
            assert g.built

        expert = FFNExpert(ffn_config={'type': 'mlp', 'hidden_dim': 16, 'output_dim': 16})
        expert.build((2, 16))
        expert.build((2, 16))  # must not raise
        assert expert.built

    # --- F2: SoftMoE raw_gate_probs contract shape -----------------------

    def test_softmoe_raw_gate_probs_shape_and_aux_loss(self):
        """raw_gate_probs must be [batch, seq, num_experts] and feed aux loss."""
        num_experts = 4
        g = SoftMoEGating(num_experts=num_experts, num_slots=3)
        x = np.random.randn(2, 5, 16).astype('float32')
        weights, _, info = g(x, training=False)

        rgp = info['raw_gate_probs']
        assert tuple(rgp.shape) == (2, 5, num_experts)

        # marginal over experts is a probability distribution
        sums = ops.convert_to_numpy(ops.sum(rgp, axis=-1))
        np.testing.assert_allclose(sums, np.ones_like(sums), atol=1e-4)

        # accepted by compute_auxiliary_loss without shape error
        loss = compute_auxiliary_loss(weights, rgp, num_experts=num_experts)
        assert np.isfinite(float(ops.convert_to_numpy(loss)))

    # --- F5: GatingConfig validation symmetry ----------------------------

    @pytest.mark.parametrize("kwargs", [
        {'gating_type': 'bogus'},
        {'top_k': 0},
        {'num_slots': 0},
        {'embedding_dim': 0},
        {'temperature': 0.0},
        {'noise_std': -1.0},
    ])
    def test_gating_config_rejects_invalid(self, kwargs):
        with pytest.raises(ValueError):
            GatingConfig(**kwargs)

    def test_gating_config_accepts_valid(self):
        GatingConfig()
        GatingConfig(gating_type='cosine', top_k=2, embedding_dim=64)
        GatingConfig(gating_type='softmoe', num_slots=8)

    # --- F6: CosineGating temperature robustness -------------------------

    def test_cosine_learnable_temperature_zero_is_finite(self):
        """A learnable temperature drifting to 0 / negative must not NaN."""
        g = CosineGating(num_experts=4, top_k=2, embedding_dim=8,
                         temperature=1.0, learnable_temperature=True)
        g.build((2, 5, 16))
        x = np.random.randn(2, 5, 16).astype('float32')
        for bad in (0.0, -0.5):
            g.temperature_param.assign(bad)
            w = ops.convert_to_numpy(g(x, training=False)[0])
            assert np.isfinite(w).all()
            np.testing.assert_allclose(w.sum(-1), np.ones(w.shape[:-1]), atol=1e-4)


class TestMoEConfigValidation:
    """F-3 / B1 + B3: the config layer owns ``num_experts``, ``top_k`` and ``jitter_noise``.

    Before step 4 these invariants were enforced *only* inside
    ``LinearGating.__init__`` / ``CosineGating.__init__``, two classes away from the
    object every real consumer constructs. The measured consequence: deleting the
    upper bound at both ``gating.py`` sites left 83 of 84 tests passing, and
    ``CosineGating``'s copy of the guard had zero coverage.

    The tests below split deliberately into two groups:

    * config-level -- reject at ``MoEConfig`` construction;
    * layer-level -- the retained ``gating.py`` guards, reached by mutating an
      already-constructed config, which is the only way past the config guard.
      These are the ones that must go RED under the F-3 mutation.
    """

    @staticmethod
    def _small_expert() -> ExpertConfig:
        """A cheap, valid expert configuration.

        :return: An ``ExpertConfig`` whose FFN is small enough to construct fast.
        :rtype: ExpertConfig
        """
        return ExpertConfig(
            ffn_config={'type': 'mlp', 'hidden_dim': 16, 'output_dim': 8}
        )

    # --- config-level rejection ------------------------------------------

    @pytest.mark.parametrize("gating_type", ['linear', 'cosine'])
    def test_top_k_above_num_experts_rejected_by_moe_config(self, gating_type):
        """``top_k > num_experts`` must fail at ``MoEConfig`` construction."""
        with pytest.raises(ValueError, match="top_k must be between 1 and num_experts"):
            MoEConfig(
                num_experts=4,
                gating_config=GatingConfig(gating_type=gating_type, top_k=8),
            )

    @pytest.mark.parametrize("bad", [0, -3])
    def test_num_experts_below_one_rejected_by_moe_config(self, bad):
        """``num_experts < 1`` must fail at ``MoEConfig`` construction."""
        with pytest.raises(ValueError, match="num_experts must be >= 1"):
            MoEConfig(num_experts=bad)

    def test_negative_jitter_noise_rejected_by_moe_config(self):
        """B3: a negative ``jitter_noise`` is REJECTED, not silently disabled."""
        with pytest.raises(ValueError, match="jitter_noise must be >= 0"):
            MoEConfig(num_experts=4, jitter_noise=-1.0)

    def test_zero_jitter_noise_is_accepted(self):
        """``jitter_noise=0`` is the documented way to disable input jitter."""
        assert MoEConfig(num_experts=4, jitter_noise=0.0).jitter_noise == 0.0

    def test_top_k_equal_to_num_experts_is_accepted(self):
        """``top_k == num_experts`` is the legal "all experts" boundary."""
        cfg = MoEConfig(num_experts=4, gating_config=GatingConfig(top_k=4))
        assert cfg.gating_config.top_k == 4

    def test_softmoe_top_k_is_not_cross_checked(self):
        """SoftMoE ignores ``top_k`` by design, so it is exempt from the check.

        ``MixtureOfExperts.__init__``'s gating allow-list forwards only
        ``num_slots`` to ``SoftMoEGating``; ``top_k`` never reaches routing. This
        test pins the exemption so a future reader does not "fix" it into a
        rejection of configurations that construct and run correctly.
        """
        cfg = MoEConfig(
            num_experts=4,
            expert_config=self._small_expert(),
            gating_config=GatingConfig(gating_type='softmoe', top_k=999, num_slots=2),
        )
        assert cfg.gating_config.top_k == 999
        # And it really does build -- the field is inert, not merely tolerated.
        MixtureOfExperts(config=cfg)

    def test_valid_config_still_constructs(self):
        """The new guard must not reject the configurations consumers ship."""
        MoEConfig(
            num_experts=8,
            expert_config=self._small_expert(),
            gating_config=GatingConfig(top_k=2),
        )

    # --- layer-level rejection (the retained gating.py guards) ------------
    #
    # These three are the F-3 mutation detectors. Deleting the `top_k > num_experts`
    # upper bound at gating.py's LinearGating and CosineGating sites must turn all
    # three RED; the config-level tests above cannot see that mutation at all.

    @pytest.mark.parametrize("gating_type", ['linear', 'cosine'])
    def test_mixture_of_experts_rejects_top_k_above_num_experts(self, gating_type):
        """``MixtureOfExperts`` still rejects a bad ``top_k`` reaching it.

        The config guard makes this unreachable through normal construction, so the
        config is mutated after the fact -- exercising the ``gating.py`` guards that
        are deliberately retained as defence in depth.
        """
        config = MoEConfig(
            num_experts=4,
            expert_config=self._small_expert(),
            gating_config=GatingConfig(
                gating_type=gating_type, top_k=2, embedding_dim=8
            ),
        )
        config.gating_config.top_k = 8  # post-construction mutation
        with pytest.raises(ValueError, match="top_k must be between 1 and 4"):
            MixtureOfExperts(config=config)

    def test_cosine_gating_rejects_top_k_above_num_experts(self):
        """``CosineGating``'s own guard -- measured to have ZERO coverage before."""
        with pytest.raises(ValueError, match="top_k must be between 1 and 4"):
            CosineGating(num_experts=4, top_k=8, embedding_dim=8)



class TestFactoryAndShapeContract:
    """D1 + D2: `compute_output_shape` honesty and `create_ffn_moe`'s keyword contract."""

    @pytest.mark.parametrize("ffn_type", ["swin_mlp", "gelu_tanh"])
    def test_explicit_none_output_dim_reports_the_input_width(self, ffn_type):
        """RED before the fix: symbolic (None, 5, None) vs a measured runtime (2, 5, 10).

        `output_dim: None` is the FFN factory's "same width as the input" spelling,
        not a declared width. Membership-testing the key propagated the None.
        """
        config = MoEConfig(
            num_experts=4,
            expert_config=ExpertConfig(
                ffn_config={"type": ffn_type, "hidden_dim": 8, "output_dim": None}
            ),
            gating_config=GatingConfig(gating_type="linear", top_k=2),
        )
        layer = MixtureOfExperts(config=config)

        symbolic = layer.compute_output_shape((None, 5, 10))
        runtime = tuple(layer(np.random.rand(2, 5, 10).astype("float32")).shape)

        assert symbolic[-1] == 10, f"last dim went symbolic-None: {symbolic}"
        assert symbolic == (None, 5, 10)
        assert symbolic[1:] == runtime[1:]

    def test_explicit_output_dim_still_overrides_the_input_width(self):
        """The None-tolerant test must not become blind to a real declared width."""
        config = MoEConfig(
            num_experts=4,
            expert_config=ExpertConfig(
                ffn_config={"type": "mlp", "hidden_dim": 8, "output_dim": 6}
            ),
            gating_config=GatingConfig(gating_type="linear", top_k=2),
        )
        layer = MixtureOfExperts(config=config)
        assert layer.compute_output_shape((None, 5, 10)) == (None, 5, 6)

    def test_gate_use_bias_reaches_the_router(self):
        """The renamed key routes where its name says: the gating Dense bias.

        Asserted at ``True``, against a ``GatingConfig.use_bias`` default of
        ``False`` -- at the default the assertion cannot fail and would have
        passed against pre-fix code that dropped the key entirely.
        """
        assert GatingConfig.use_bias is False, "test is vacuous unless the default is False"
        layer = create_ffn_moe(
            num_experts=4,
            ffn_config={"type": "mlp", "hidden_dim": 8, "output_dim": 6},
            top_k=2,
            gate_use_bias=True,
        )
        assert layer.config.gating_config.use_bias is True
        # and it must not leak into the expert FFN, which is the other real route
        assert "use_bias" not in layer.config.expert_config.ffn_config

    def test_bare_use_bias_is_rejected_and_names_both_routes(self):
        """RED before the fix: `use_bias=False` was silently applied to the GATE.

        A caller passing it alongside an `ffn_config` means the expert FFN's bias,
        and got the router's instead with no error.
        """
        with pytest.raises(ValueError) as excinfo:
            create_ffn_moe(
                num_experts=4,
                ffn_config={"type": "mlp", "hidden_dim": 8, "output_dim": 6},
                use_bias=False,
            )
        message = str(excinfo.value)
        assert "use_bias" in message
        assert "gate_use_bias" in message
        assert "ffn_config['use_bias']" in message

    def test_undeclared_keyword_raises_instead_of_being_dropped(self):
        """The repo factory contract (layers/CLAUDE.md rule 1): never filter-and-drop."""
        with pytest.raises(ValueError, match="undeclared keyword"):
            create_ffn_moe(
                num_experts=4,
                ffn_config={"type": "mlp", "hidden_dim": 8, "output_dim": 6},
                bogus_key=1,
            )

    def test_keras_layer_keywords_are_forwarded(self):
        """`name=` was among the silently dropped keys; the README's example used it."""
        layer = create_ffn_moe(
            num_experts=4,
            ffn_config={"type": "mlp", "hidden_dim": 8, "output_dim": 6},
            top_k=2,
            name="moe_ffn",
        )
        assert layer.name == "moe_ffn"



class TestIntegerFieldsRejectBool:
    """B2 / F-13: `bool` is an `int` subclass, so every int field silently took it.

    RED-proven at 1ac2908e7: `GatingConfig(top_k=True)` constructed with
    `top_k=True` (arithmetic value 1), `GatingConfig(embedding_dim=True)` gave a
    one-dimensional expert embedding, and `MoEConfig(num_experts=True)` gave a
    one-expert MoE. YAML is the live path -- `yaml.safe_load("top_k: true")`
    returns Python `True`.
    """

    @pytest.mark.parametrize("value", [True, False])
    @pytest.mark.parametrize("field", ["top_k", "num_slots", "embedding_dim"])
    def test_gating_config_int_fields_reject_bool(self, field, value):
        with pytest.raises(ValueError, match=f"{field} must be an int, got bool"):
            GatingConfig(**{field: value})

    @pytest.mark.parametrize("value", [True, False])
    def test_moe_config_num_experts_rejects_bool(self, value):
        with pytest.raises(ValueError, match="num_experts must be an int, got bool"):
            MoEConfig(num_experts=value)

    def test_yaml_true_is_the_live_path(self):
        """The value a config-driven caller actually gets from an unquoted `true`."""
        yaml = pytest.importorskip("yaml")
        loaded = yaml.safe_load("top_k: true")["top_k"]
        assert loaded is True
        with pytest.raises(ValueError, match="must be an int, got bool"):
            GatingConfig(top_k=loaded)

    @pytest.mark.parametrize("field", ["top_k", "num_slots", "embedding_dim"])
    def test_non_int_types_are_rejected(self, field):
        with pytest.raises(ValueError, match=f"{field} must be an int, got"):
            GatingConfig(**{field: 2.5})

    def test_real_ints_still_construct(self):
        """The bool branch must not have swallowed the ordinary path."""
        gating = GatingConfig(top_k=3, num_slots=5, embedding_dim=64)
        assert (gating.top_k, gating.num_slots, gating.embedding_dim) == (3, 5, 64)
        assert MoEConfig(num_experts=4).num_experts == 4

    @pytest.mark.parametrize(
        "field,message",
        [
            ("top_k", "top_k must be >= 1, got 0"),
            ("num_slots", "num_slots must be >= 1, got 0"),
            ("embedding_dim", "embedding_dim must be >= 1, got 0"),
        ],
    )
    def test_range_messages_are_unchanged(self, field, message):
        """The bool check is inserted ahead of the range check, not in place of it."""
        with pytest.raises(ValueError, match=message):
            GatingConfig(**{field: 0})


# Run tests with: pytest test_mixture_of_experts.py -v
if __name__ == "__main__":
    pytest.main([__file__, "-v"])