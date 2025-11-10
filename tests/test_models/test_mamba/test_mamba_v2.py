"""
Comprehensive pytest test suite for the Mamba v2 foundation model.

This module provides extensive testing for the Mamba v2 implementation,
mirroring the structure of the Mamba v1 test suite. It covers:
- Foundation model initialization and V2-specific parameter validation.
- Architecture building and consistent output shape.
- Forward pass functionality, including the parallel SSM/MLP paths.
- Model variant creation and configuration for V2 architectures.
- Serialization and deserialization of the foundation model.
- Error handling and edge cases.
- End-to-end integration testing for gradient flow and training.
- Advanced V2-specific features like RMSNorm toggling.
"""

import pytest
import numpy as np
import keras
import tensorflow as tf
import tempfile
import os
from typing import Dict, Any

from dl_techniques.models.mamba.mamba_v2 import Mamba2
from dl_techniques.models.mamba.components_v2 import Mamba2Layer

class TestMamba2ModelInitialization:
    """Test Mamba v2 model initialization and parameter validation."""

    def test_basic_initialization(self):
        """Test basic Mamba v2 model initialization as a pure encoder."""
        model = Mamba2(
            vocab_size=1000,
            d_model=256,
            num_layers=6,
            d_state=128,
            d_conv=4,
            expand=2,
            headdim=64
        )
        assert model.d_model == 256
        assert model.num_layers == 6
        assert model.d_state == 128
        assert model.headdim == 64
        assert not model.built
        assert len(model.encoder_layers) == 6

    def test_parameter_validation(self):
        """Test Mamba v2 parameter validation for invalid values."""
        with pytest.raises(ValueError, match="vocab_size must be positive"):
            Mamba2(vocab_size=0, d_model=256, num_layers=4)
        with pytest.raises(ValueError, match="vocab_size must be positive"):
            Mamba2(vocab_size=-1, d_model=256, num_layers=4)
        with pytest.raises(ValueError, match="d_model must be positive"):
            Mamba2(vocab_size=1000, d_model=0, num_layers=4)
        with pytest.raises(ValueError, match="num_layers must be positive"):
            Mamba2(vocab_size=1000, d_model=256, num_layers=0)

        # V2 specific validation: headdim must divide d_ssm
        with pytest.raises(ValueError, match="d_ssm .* must be divisible by headdim"):
            d_ssm = 256 * 2  # d_inner
            # This layer will fail on build because 512 is not divisible by 60
            Mamba2Layer(d_model=256, expand=2, headdim=60)

    def test_initialization_with_custom_config(self):
        """Test Mamba v2 model initialization with custom configuration."""
        model = Mamba2(
            vocab_size=25000,
            d_model=512,
            num_layers=8,
            d_state=64,
            d_conv=8,
            expand=3,
            headdim=128,
            norm_epsilon=1e-6,
            rmsnorm=False
        )
        assert model.rmsnorm is False
        assert model.norm_epsilon == 1e-6
        # Check a property of the inner layer
        first_mamba2_layer = model.encoder_layers[0].mamba2
        assert first_mamba2_layer.rmsnorm is False


class TestMamba2ModelVariants:
    """Test Mamba v2 model variants and factory methods."""

    @pytest.mark.parametrize("variant, d_model, num_layers", [
        ("130m", 768, 24),
        ("base", 768, 24),
        ("370m", 1024, 24),
        ("780m", 1536, 36),
        ("1.4b", 2048, 48),
        ("2.8b", 2560, 64),
    ])
    def test_variants(self, variant, d_model, num_layers):
        """Test all standard parameter variants."""
        model = Mamba2.from_variant(variant, vocab_size=50257)
        assert model.d_model == d_model
        assert model.num_layers == num_layers
        assert model.vocab_size == 50257

    def test_invalid_variant(self):
        """Test error handling for invalid variant names."""
        with pytest.raises(ValueError, match="Unknown variant 'invalid'"):
            Mamba2.from_variant("invalid", vocab_size=50257)

    def test_variant_with_custom_params(self):
        """Test creating variant with custom parameter overrides."""
        model = Mamba2.from_variant(
            "base",
            vocab_size=50257,
            d_state=64,
            expand=3
        )
        assert model.d_model == 768
        assert model.num_layers == 24
        assert model.d_state == 64
        assert model.expand == 3


class TestMamba2ModelBuilding:
    """Test Mamba v2 model building and architecture creation."""

    @pytest.fixture
    def basic_config(self) -> Dict[str, Any]:
        return {
            "vocab_size": 1000, "d_model": 128, "num_layers": 2,
            "d_state": 32, "d_conv": 4, "expand": 2, "headdim": 32
        }

    def test_build_basic_functionality(self, basic_config):
        """Test basic building functionality and output contract."""
        model = Mamba2(**basic_config)
        input_ids = keras.random.randint((2, 16), 0, 1000)
        outputs = model({"input_ids": input_ids})

        assert model.built
        assert "last_hidden_state" in outputs
        assert outputs["last_hidden_state"].shape == (2, 16, 128)

    def test_encoder_layers_configuration(self, basic_config):
        """Test that encoder layers are properly configured."""
        model = Mamba2(**basic_config)
        _ = model(keras.random.randint((1, 8), 0, 1000))

        for i, block in enumerate(model.encoder_layers):
            assert block.d_model == 128
            assert block.mamba2.d_state == 32
            assert block.mamba2.headdim == 32


class TestMamba2ModelForwardPass:
    """Test Mamba v2 model forward pass functionality."""

    @pytest.fixture
    def built_model(self) -> Mamba2:
        model = Mamba2(
            vocab_size=1000, d_model=64, num_layers=2, d_state=16,
            d_conv=4, expand=2, headdim=32
        )
        _ = model(keras.random.randint((1, 8), 0, 1000))
        return model

    def test_forward_pass_with_tensor_input(self, built_model):
        input_ids = keras.random.randint((2, 16), 0, 1000)
        outputs = built_model(input_ids)
        assert "last_hidden_state" in outputs
        assert outputs["last_hidden_state"].shape == (2, 16, 64)

    def test_forward_pass_with_dict_input(self, built_model):
        inputs = {'input_ids': keras.random.randint((3, 12), 0, 1000)}
        outputs = built_model(inputs)
        assert "last_hidden_state" in outputs
        assert outputs["last_hidden_state"].shape == (3, 12, 64)

    def test_forward_pass_variable_sequence_lengths(self, built_model):
        for seq_length in [8, 16, 32]:
            outputs = built_model(keras.random.randint((2, seq_length), 0, 1000))
            assert outputs["last_hidden_state"].shape == (2, seq_length, 64)

    def test_invalid_dict_input(self, built_model):
        with pytest.raises(ValueError, match="Dictionary input must contain 'input_ids' key"):
            built_model({'invalid_key': keras.ops.ones((2, 16), 'int32')})

    def test_training_vs_inference_mode(self, built_model):
        input_ids = keras.random.randint((2, 16), 0, 1000)
        output_train = built_model(input_ids, training=True)
        assert output_train["last_hidden_state"].shape == (2, 16, 64)

        output_inference1 = built_model(input_ids, training=False)
        output_inference2 = built_model(input_ids, training=False)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(output_inference1["last_hidden_state"]),
            keras.ops.convert_to_numpy(output_inference2["last_hidden_state"])
        )

class TestMamba2ModelSerialization:
    """Test Mamba v2 model serialization and deserialization."""

    def test_config_serialization(self):
        model = Mamba2(
            vocab_size=1000, d_model=128, num_layers=2, d_state=32, headdim=64
        )
        config = model.get_config()
        assert config['vocab_size'] == 1000
        assert config['d_model'] == 128
        assert config['headdim'] == 64

    def test_model_save_load_cycle(self):
        model = Mamba2(vocab_size=1000, d_model=64, num_layers=2, d_state=16, headdim=32)
        input_ids = keras.random.randint((2, 16), 0, 1000)
        original_outputs = model(input_ids)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "mamba2.keras")
            model.save(path)
            loaded_model = keras.models.load_model(path)
            loaded_outputs = loaded_model(input_ids)

            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_outputs['last_hidden_state']),
                keras.ops.convert_to_numpy(loaded_outputs['last_hidden_state']),
                rtol=1e-5
            )

class TestMamba2EdgeCases:
    """Test Mamba v2 model edge cases."""

    def test_minimum_sequence_length(self):
        model = Mamba2(vocab_size=1000, d_model=64, num_layers=2, d_state=16, headdim=32)
        outputs = model(keras.ops.array([[42]], "int32"))
        assert outputs['last_hidden_state'].shape == (1, 1, 64)

    def test_mlp_path_integration(self):
        """Test model where d_ssm is less than d_inner, activating MLP path."""
        d_model = 128
        expand = 2
        d_inner = d_model * expand
        d_ssm = d_inner // 2

        model = Mamba2(
            vocab_size=1000, d_model=d_model, num_layers=2, d_state=32,
            expand=expand, headdim=32, d_ssm=d_ssm
        )

        outputs = model(keras.random.randint((2, 16), 0, 1000))
        assert outputs['last_hidden_state'].shape == (2, 16, d_model)
        assert model.encoder_layers[0].mamba2.d_ssm == d_ssm

    def test_output_not_nan_or_inf(self):
        """Test that outputs don't contain NaN or Inf values."""
        model = Mamba2(vocab_size=1000, d_model=64, num_layers=2, d_state=16, headdim=32)
        outputs = model(keras.random.randint((4, 32), 0, 1000))
        hidden_states = keras.ops.convert_to_numpy(outputs['last_hidden_state'])
        assert not np.isnan(hidden_states).any()
        assert not np.isinf(hidden_states).any()


class TestMamba2Integration:
    """Integration tests for the complete Mamba v2 model."""

    @pytest.fixture
    def small_model(self) -> Mamba2:
        return Mamba2(
            vocab_size=1000, d_model=64, num_layers=2, d_state=16,
            d_conv=2, expand=2, headdim=32
        )

    def test_gradient_flow(self, small_model):
        """Test that gradients flow through the entire model."""
        input_ids = keras.random.randint((2, 16), 0, 1000)
        with tf.GradientTape() as tape:
            outputs = small_model(input_ids, training=True)
            loss = keras.ops.mean(outputs['last_hidden_state']**2)
        gradients = tape.gradient(loss, small_model.trainable_weights)

        assert all(g is not None for g in gradients)
        assert all(keras.ops.any(g != 0) for g in gradients)

    def test_gradient_flow_through_ssm_params(self, small_model):
        """Test gradients for A_log, D, and dt_bias."""
        input_ids = keras.random.randint((2, 16), 0, 1000)
        with tf.GradientTape() as tape:
            outputs = small_model(input_ids, training=True)
            loss = keras.ops.mean(outputs['last_hidden_state']**2)

        mamba_layer = small_model.encoder_layers[0].mamba2
        ssm_params = [mamba_layer.A_log, mamba_layer.D, mamba_layer.dt_bias]
        grads = tape.gradient(loss, ssm_params)

        for grad in grads:
            assert grad is not None
            assert not np.allclose(keras.ops.convert_to_numpy(grad), 1e-1)

    def test_training_integration(self, small_model):
        """Test the model in a minimal training loop."""
        optimizer = keras.optimizers.Adam()
        input_ids = keras.random.randint((4, 16), 0, 1000)
        targets = keras.random.randint((4, 16), 0, 1000)

        with tf.GradientTape() as tape:
            outputs = small_model(input_ids, training=True)
            logits = keras.layers.Dense(1000)(outputs['last_hidden_state'])
            initial_loss = keras.losses.sparse_categorical_crossentropy(
                targets, logits, from_logits=True
            )
        grads = tape.gradient(initial_loss, small_model.trainable_weights)
        optimizer.apply_gradients(zip(grads, small_model.trainable_weights))

        with tf.GradientTape() as tape:
            outputs = small_model(input_ids, training=True)
            logits = keras.layers.Dense(1000)(outputs['last_hidden_state'])
            final_loss = keras.losses.sparse_categorical_crossentropy(
                targets, logits, from_logits=True
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])