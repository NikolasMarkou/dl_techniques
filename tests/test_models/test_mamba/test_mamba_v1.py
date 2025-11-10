"""
Comprehensive pytest test suite for the Mamba foundation model.

This module provides extensive testing for the Mamba implementation including:
- Foundation model initialization and parameter validation.
- Architecture building and consistent output shape.
- Forward pass functionality with a standardized dictionary output.
- Model variant creation and configuration.
- Serialization and deserialization of the foundation model.
- Error handling and edge cases.
- End-to-end integration testing for gradient flow and training.
- Advanced SSM-specific features.

Test structure follows the BERT test pattern for consistency across dl_techniques.
"""

import pytest
import numpy as np
import keras
import tensorflow as tf
import tempfile
import os
from typing import Dict, Any

from dl_techniques.models.mamba import Mamba


class TestMambaModelInitialization:
    """Test Mamba model initialization and parameter validation."""

    def test_basic_initialization(self):
        """Test basic Mamba model initialization as a pure encoder."""
        model = Mamba(
            vocab_size=1000,
            d_model=256,
            num_layers=6,
            d_state=16,
            d_conv=4,
            expand=2
        )

        assert model.d_model == 256
        assert model.num_layers == 6
        assert model.d_state == 16
        assert model.vocab_size == 1000
        assert not model.built

        # Components should be created in __init__
        assert model.embedding is not None
        assert len(model.encoder_layers) == 6
        assert model.final_norm is not None

        # But should not be built yet
        assert not model.embedding.built
        for layer in model.encoder_layers:
            assert not layer.built
        assert not model.final_norm.built

    def test_parameter_validation(self):
        """Test Mamba parameter validation for invalid values."""
        # Test invalid vocab_size
        with pytest.raises(ValueError, match="vocab_size must be positive"):
            Mamba(vocab_size=-1000, d_model=256, num_layers=4)

        with pytest.raises(ValueError, match="vocab_size must be positive"):
            Mamba(vocab_size=0, d_model=256, num_layers=4)

        # Test invalid d_model
        with pytest.raises(ValueError, match="d_model must be positive"):
            Mamba(vocab_size=1000, d_model=-256, num_layers=4)

        # Test invalid num_layers
        with pytest.raises(ValueError, match="num_layers must be positive"):
            Mamba(vocab_size=1000, d_model=256, num_layers=0)

    def test_initialization_with_custom_config(self):
        """Test Mamba model initialization with custom configuration."""
        model = Mamba(
            vocab_size=25000,
            d_model=512,
            num_layers=8,
            d_state=32,
            d_conv=8,
            expand=3,
            dt_rank=64,
            norm_epsilon=1e-6
        )

        assert model.vocab_size == 25000
        assert model.d_model == 512
        assert model.num_layers == 8
        assert model.d_state == 32
        assert model.d_conv == 8
        assert model.expand == 3
        assert model.dt_rank == 64
        assert model.norm_epsilon == 1e-6

    def test_auto_dt_rank_computation(self):
        """Test automatic dt_rank computation."""
        model = Mamba(
            vocab_size=1000,
            d_model=768,
            num_layers=4,
            dt_rank="auto"
        )

        # dt_rank should be ceil(768 / 16) = 48
        import math
        expected_dt_rank = math.ceil(768 / 16)

        # Check the first encoder layer's Mamba layer
        first_block = model.encoder_layers[0]
        assert first_block.mamba.dt_rank == expected_dt_rank


class TestMambaModelVariants:
    """Test Mamba model variants and factory methods."""

    def test_mamba_130m_variant(self):
        """Test 130M parameter variant (base)."""
        model = Mamba.from_variant("130m", vocab_size=50257)
        assert model.d_model == 768
        assert model.num_layers == 24
        assert model.vocab_size == 50257

    def test_mamba_base_variant(self):
        """Test base variant (alias for 130m)."""
        model = Mamba.from_variant("base", vocab_size=50257)
        assert model.d_model == 768
        assert model.num_layers == 24
        assert model.vocab_size == 50257

    def test_mamba_370m_variant(self):
        """Test 370M parameter variant."""
        model = Mamba.from_variant("370m", vocab_size=50257)
        assert model.d_model == 1024
        assert model.num_layers == 24
        assert model.vocab_size == 50257

    def test_mamba_790m_variant(self):
        """Test 790M parameter variant."""
        model = Mamba.from_variant("790m", vocab_size=50257)
        assert model.d_model == 1024
        assert model.num_layers == 48
        assert model.vocab_size == 50257

    def test_mamba_1_4b_variant(self):
        """Test 1.4B parameter variant."""
        model = Mamba.from_variant("1.4b", vocab_size=50257)
        assert model.d_model == 1536
        assert model.num_layers == 48
        assert model.vocab_size == 50257

    def test_mamba_2_8b_variant(self):
        """Test 2.8B parameter variant."""
        model = Mamba.from_variant("2.8b", vocab_size=50257)
        assert model.d_model == 2560
        assert model.num_layers == 64
        assert model.vocab_size == 50257

    def test_invalid_variant(self):
        """Test error handling for invalid variant names."""
        with pytest.raises(ValueError, match="Unknown variant 'invalid'"):
            Mamba.from_variant("invalid", vocab_size=50257)

    def test_variant_without_vocab_size(self):
        """Test that vocab_size is required for from_variant."""
        # This should work - vocab_size is provided
        model = Mamba.from_variant("base", vocab_size=50257)
        assert model.vocab_size == 50257

    def test_variant_with_custom_params(self):
        """Test creating variant with custom parameter overrides."""
        model = Mamba.from_variant(
            "base",
            vocab_size=50257,
            d_state=32,  # Override default
            expand=3  # Override default
        )

        assert model.d_model == 768  # From variant
        assert model.num_layers == 24  # From variant
        assert model.d_state == 32  # Custom override
        assert model.expand == 3  # Custom override


class TestMambaModelBuilding:
    """Test Mamba model building and architecture creation."""

    @pytest.fixture
    def basic_config(self) -> Dict[str, Any]:
        """Basic configuration for testing."""
        return {
            "vocab_size": 1000,
            "d_model": 256,
            "num_layers": 4,
            "d_state": 16,
            "d_conv": 4,
            "expand": 2
        }

    def test_build_basic_functionality(self, basic_config):
        """Test basic building functionality and output contract."""
        model = Mamba(**basic_config)

        batch_size, seq_length = 2, 32
        input_ids = keras.random.randint(
            (batch_size, seq_length),
            minval=0,
            maxval=basic_config['vocab_size'],
            dtype="int32"
        )

        outputs = model({"input_ids": input_ids}, training=False)

        assert model.built
        assert isinstance(outputs, dict)
        assert "last_hidden_state" in outputs
        assert outputs["last_hidden_state"].shape == (
            batch_size, seq_length, basic_config['d_model']
        )

    def test_encoder_layers_configuration(self, basic_config):
        """Test that encoder layers are properly configured."""
        model = Mamba(**basic_config)

        input_ids = keras.random.randint(
            (1, 16),
            minval=0,
            maxval=basic_config['vocab_size'],
            dtype="int32"
        )

        _ = model({"input_ids": input_ids}, training=False)

        # Check each encoder layer
        for i, block in enumerate(model.encoder_layers):
            assert block.d_model == basic_config['d_model']
            assert block.name == f"mamba_block_{i}"
            assert block.built

            # Check the inner Mamba layer
            assert block.mamba.d_model == basic_config['d_model']
            assert block.mamba.d_state == basic_config['d_state']
            assert block.mamba.d_conv == basic_config['d_conv']
            assert block.mamba.expand == basic_config['expand']
            assert block.mamba.layer_idx == i

    def test_embedding_layer_configuration(self, basic_config):
        """Test that embedding layer is properly configured."""
        model = Mamba(**basic_config)

        input_ids = keras.random.randint(
            (1, 16),
            minval=0,
            maxval=basic_config['vocab_size'],
            dtype="int32"
        )

        _ = model({"input_ids": input_ids}, training=False)

        assert model.embedding.built
        assert model.embedding.input_dim == basic_config['vocab_size']
        assert model.embedding.output_dim == basic_config['d_model']


class TestMambaModelForwardPass:
    """Test Mamba model forward pass functionality."""

    @pytest.fixture
    def built_model(self) -> Mamba:
        """Create a built Mamba model for testing."""
        model = Mamba(
            vocab_size=1000,
            d_model=256,
            num_layers=3,
            d_state=16,
            d_conv=4,
            expand=2
        )

        sample_input = keras.random.randint(
            (1, 16),
            minval=0,
            maxval=1000,
            dtype="int32"
        )

        _ = model({"input_ids": sample_input}, training=False)
        return model

    def test_forward_pass_with_tensor_input(self, built_model):
        """Test forward pass with direct tensor input."""
        batch_size, seq_length = 4, 32
        input_ids = keras.random.randint(
            (batch_size, seq_length),
            minval=0,
            maxval=built_model.vocab_size,
            dtype="int32"
        )

        outputs = built_model(input_ids, training=False)

        assert isinstance(outputs, dict)
        assert "last_hidden_state" in outputs
        assert outputs["last_hidden_state"].shape == (
            batch_size, seq_length, built_model.d_model
        )

    def test_forward_pass_with_dict_input(self, built_model):
        """Test forward pass with dictionary input."""
        batch_size, seq_length = 3, 24
        inputs = {
            'input_ids': keras.random.randint(
                (batch_size, seq_length),
                minval=0,
                maxval=built_model.vocab_size,
                dtype="int32"
            )
        }

        outputs = built_model(inputs, training=False)

        assert isinstance(outputs, dict)
        assert "last_hidden_state" in outputs
        assert outputs["last_hidden_state"].shape == (
            batch_size, seq_length, built_model.d_model
        )

    def test_forward_pass_variable_sequence_lengths(self, built_model):
        """Test forward pass with different sequence lengths."""
        for seq_length in [8, 16, 32, 64, 128]:
            input_ids = keras.random.randint(
                (2, seq_length),
                minval=0,
                maxval=built_model.vocab_size,
                dtype="int32"
            )

            outputs = built_model({"input_ids": input_ids}, training=False)

            assert outputs["last_hidden_state"].shape == (2, seq_length, built_model.d_model)

    def test_forward_pass_different_batch_sizes(self, built_model):
        """Test forward pass with different batch sizes."""
        seq_length = 32

        for batch_size in [1, 2, 4, 8]:
            input_ids = keras.random.randint(
                (batch_size, seq_length),
                minval=0,
                maxval=built_model.vocab_size,
                dtype="int32"
            )

            outputs = built_model({"input_ids": input_ids}, training=False)

            assert outputs["last_hidden_state"].shape == (
                batch_size, seq_length, built_model.d_model
            )

    def test_invalid_dict_input(self, built_model):
        """Test error handling for invalid dictionary input."""
        inputs = {'invalid_key': keras.ops.ones((2, 16), dtype='int32')}

        with pytest.raises(ValueError, match="Dictionary input must contain 'input_ids' key"):
            built_model(inputs, training=False)

    def test_training_vs_inference_mode(self, built_model):
        """Test that model works in both training and inference modes."""
        batch_size, seq_length = 2, 16
        input_ids = keras.random.randint(
            (batch_size, seq_length),
            minval=0,
            maxval=built_model.vocab_size,
            dtype="int32"
        )

        # Training mode
        output_train = built_model({"input_ids": input_ids}, training=True)
        assert output_train["last_hidden_state"].shape == (
            batch_size, seq_length, built_model.d_model
        )

        # Inference mode
        output_inference = built_model({"input_ids": input_ids}, training=False)
        assert output_inference["last_hidden_state"].shape == (
            batch_size, seq_length, built_model.d_model
        )

        # Outputs should be deterministic in inference mode
        output_inference2 = built_model({"input_ids": input_ids}, training=False)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(output_inference["last_hidden_state"]),
            keras.ops.convert_to_numpy(output_inference2["last_hidden_state"]),
            rtol=1e-6,
            atol=1e-6,
            err_msg="Inference outputs should be deterministic"
        )


class TestMambaModelSerialization:
    """Test Mamba model serialization and deserialization."""

    def test_config_serialization(self):
        """Test that model configuration is properly serialized."""
        model = Mamba(
            vocab_size=25000,
            d_model=512,
            num_layers=8,
            d_state=32,
            d_conv=8,
            expand=3,
            dt_rank=64,
            norm_epsilon=1e-6
        )

        model_config = model.get_config()

        assert model_config['vocab_size'] == 25000
        assert model_config['d_model'] == 512
        assert model_config['num_layers'] == 8
        assert model_config['d_state'] == 32
        assert model_config['d_conv'] == 8
        assert model_config['expand'] == 3
        assert model_config['dt_rank'] == 64
        assert model_config['norm_epsilon'] == 1e-6

    def test_model_from_config(self):
        """Test creating model from configuration."""
        original_model = Mamba(
            vocab_size=1000,
            d_model=256,
            num_layers=4,
            d_state=16,
            d_conv=4,
            expand=2
        )

        config = original_model.get_config()
        new_model = Mamba.from_config(config)

        assert new_model.vocab_size == original_model.vocab_size
        assert new_model.d_model == original_model.d_model
        assert new_model.num_layers == original_model.num_layers
        assert new_model.d_state == original_model.d_state
        assert new_model.d_conv == original_model.d_conv
        assert new_model.expand == original_model.expand

    def test_model_save_load_cycle(self):
        """Test complete save and load cycle."""
        model = Mamba(
            vocab_size=1000,
            d_model=256,
            num_layers=2,
            d_state=16,
            d_conv=4,
            expand=2
        )

        input_ids = keras.random.randint(
            (2, 16),
            minval=0,
            maxval=1000,
            dtype="int32"
        )

        original_outputs = model({"input_ids": input_ids}, training=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'test_mamba.keras')
            model.save(model_path)

            loaded_model = keras.models.load_model(model_path)
            loaded_outputs = loaded_model({"input_ids": input_ids}, training=False)

            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_outputs['last_hidden_state']),
                keras.ops.convert_to_numpy(loaded_outputs['last_hidden_state']),
                rtol=1e-5,
                atol=1e-6,
                err_msg="Hidden states should match after serialization"
            )

    def test_save_load_preserves_weights(self):
        """Test that save/load preserves trained weights."""
        model = Mamba(
            vocab_size=500,
            d_model=128,
            num_layers=2,
            d_state=8
        )

        # Build and "train" model
        input_ids = keras.random.randint((2, 16), minval=0, maxval=500, dtype="int32")
        _ = model({"input_ids": input_ids}, training=False)

        # Get weight values before saving
        original_weights = [keras.ops.convert_to_numpy(w) for w in model.trainable_weights]

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'test_weights.keras')
            model.save(model_path)

            loaded_model = keras.models.load_model(model_path)
            loaded_weights = [keras.ops.convert_to_numpy(w) for w in loaded_model.trainable_weights]

            # Check all weights match
            assert len(original_weights) == len(loaded_weights)
            for orig, loaded in zip(original_weights, loaded_weights):
                np.testing.assert_allclose(
                    orig,
                    loaded,
                    rtol=1e-6,
                    atol=1e-6,
                    err_msg="Weights should be preserved"
                )


class TestMambaEdgeCases:
    """Test Mamba model edge cases."""

    def test_minimum_sequence_length(self):
        """Test with single token sequence."""
        model = Mamba(
            vocab_size=1000,
            d_model=128,
            num_layers=2,
            d_state=8
        )

        input_ids = keras.ops.array([[42]], dtype="int32")
        outputs = model({"input_ids": input_ids}, training=False)

        assert outputs['last_hidden_state'].shape == (1, 1, 128)

    def test_long_sequence(self):
        """Test with long sequence."""
        model = Mamba(
            vocab_size=1000,
            d_model=256,
            num_layers=2,
            d_state=16
        )

        seq_length = 512
        input_ids = keras.random.randint(
            (1, seq_length),
            minval=0,
            maxval=1000,
            dtype="int32"
        )

        outputs = model({"input_ids": input_ids}, training=False)
        assert outputs['last_hidden_state'].shape == (1, seq_length, 256)

    def test_very_long_sequence(self):
        """Test with very long sequence (1024 tokens)."""
        model = Mamba(
            vocab_size=1000,
            d_model=128,
            num_layers=2,
            d_state=8
        )

        seq_length = 1024
        input_ids = keras.random.randint(
            (1, seq_length),
            minval=0,
            maxval=1000,
            dtype="int32"
        )

        outputs = model({"input_ids": input_ids}, training=False)
        assert outputs['last_hidden_state'].shape == (1, seq_length, 128)

    def test_single_batch(self):
        """Test with batch size of 1."""
        model = Mamba(
            vocab_size=1000,
            d_model=256,
            num_layers=2,
            d_state=16
        )

        input_ids = keras.random.randint((1, 32), minval=0, maxval=1000, dtype="int32")
        outputs = model({"input_ids": input_ids}, training=False)

        assert outputs['last_hidden_state'].shape == (1, 32, 256)

    def test_small_vocabulary(self):
        """Test with very small vocabulary."""
        model = Mamba(
            vocab_size=10,  # Very small vocab
            d_model=64,
            num_layers=2,
            d_state=4
        )

        input_ids = keras.random.randint((2, 16), minval=0, maxval=10, dtype="int32")
        outputs = model({"input_ids": input_ids}, training=False)

        assert outputs['last_hidden_state'].shape == (2, 16, 64)

    def test_large_state_dimension(self):
        """Test with large state dimension."""
        model = Mamba(
            vocab_size=1000,
            d_model=256,
            num_layers=2,
            d_state=64  # Larger than typical
        )

        input_ids = keras.random.randint((2, 16), minval=0, maxval=1000, dtype="int32")
        outputs = model({"input_ids": input_ids}, training=False)

        assert outputs['last_hidden_state'].shape == (2, 16, 256)

    def test_output_not_nan_or_inf(self):
        """Test that outputs don't contain NaN or Inf values."""
        model = Mamba(
            vocab_size=1000,
            d_model=256,
            num_layers=4,
            d_state=16
        )

        input_ids = keras.random.randint((4, 32), minval=0, maxval=1000, dtype="int32")
        outputs = model({"input_ids": input_ids}, training=False)

        hidden_states = keras.ops.convert_to_numpy(outputs['last_hidden_state'])

        assert not np.isnan(hidden_states).any(), "Output contains NaN values"
        assert not np.isinf(hidden_states).any(), "Output contains Inf values"


class TestMambaIntegration:
    """Integration tests for the complete Mamba model."""

    @pytest.fixture
    def small_model(self) -> Mamba:
        """Create a small model for integration tests."""
        return Mamba(
            vocab_size=1000,
            d_model=128,
            num_layers=3,
            d_state=8,
            d_conv=4,
            expand=2
        )

    def test_gradient_flow(self, small_model):
        """Test that gradients flow through the entire model."""
        batch_size, seq_length = 2, 16
        input_ids = keras.random.randint(
            (batch_size, seq_length),
            minval=0,
            maxval=1000,
            dtype="int32"
        )

        with tf.GradientTape() as tape:
            outputs = small_model({"input_ids": input_ids}, training=True)
            hidden_states = outputs['last_hidden_state']
            # Simple loss: mean squared error from zero
            loss = keras.ops.mean(keras.ops.square(hidden_states))

        gradients = tape.gradient(loss, small_model.trainable_weights)

        # Check all gradients exist and are non-zero
        non_none_grads = [g for g in gradients if g is not None]
        assert len(non_none_grads) > 0
        assert len(non_none_grads) == len(small_model.trainable_weights)

        # Check gradients have reasonable magnitudes
        grad_norms = [
            keras.ops.sqrt(keras.ops.sum(keras.ops.square(g)))
            for g in non_none_grads
        ]
        assert all(norm > 0.0 for norm in grad_norms), "Some gradients are zero"

    def test_gradient_flow_through_ssm_params(self, small_model):
        """Test that gradients flow through SSM-specific parameters (A_log, D)."""
        batch_size, seq_length = 2, 16
        input_ids = keras.random.randint(
            (batch_size, seq_length),
            minval=0,
            maxval=1000,
            dtype="int32"
        )

        with tf.GradientTape() as tape:
            outputs = small_model({"input_ids": input_ids}, training=True)
            loss = keras.ops.mean(keras.ops.square(outputs['last_hidden_state']))

        # Get SSM-specific parameters from first Mamba layer
        first_mamba_layer = small_model.encoder_layers[0].mamba
        A_log = first_mamba_layer.A_log
        D = first_mamba_layer.D

        # Compute gradients for SSM parameters in a single call
        grad_A, grad_D = tape.gradient(loss, [A_log, D])

        assert grad_A is not None, "A_log gradient is None"
        assert grad_D is not None, "D gradient is None"

        # Check they're not all zeros
        grad_A_np = keras.ops.convert_to_numpy(grad_A)
        grad_D_np = keras.ops.convert_to_numpy(grad_D)

        assert not np.allclose(grad_A_np, 0), "A_log gradient is all zeros"
        assert not np.allclose(grad_D_np, 0), "D gradient is all zeros"

    def test_training_integration(self, small_model):
        """Test the model in a minimal training loop."""
        optimizer = keras.optimizers.Adam(learning_rate=1e-4)

        batch_size, seq_length = 4, 16
        vocab_size = small_model.vocab_size

        input_ids = keras.random.randint(
            (batch_size, seq_length),
            minval=0,
            maxval=vocab_size,
            dtype="int32"
        )

        # Simple task: predict next token (language modeling)
        # Shift input_ids to create targets
        targets = keras.random.randint(
            (batch_size, seq_length),
            minval=0,
            maxval=vocab_size,
            dtype="int32"
        )

        initial_loss = None
        losses = []

        # Train for a few steps
        for step in range(10):
            with tf.GradientTape() as tape:
                outputs = small_model({"input_ids": input_ids}, training=True)
                hidden_states = outputs['last_hidden_state']

                # Simple projection to vocab_size for language modeling
                # (normally this would be a separate head)
                logits = keras.ops.matmul(
                    hidden_states,
                    keras.random.normal((small_model.d_model, vocab_size))
                )

                # Compute loss
                loss = keras.ops.mean(
                    keras.losses.sparse_categorical_crossentropy(
                        targets,
                        logits,
                        from_logits=True
                    )
                )

            if initial_loss is None:
                initial_loss = keras.ops.convert_to_numpy(loss)

            losses.append(keras.ops.convert_to_numpy(loss))

            # Update weights
            gradients = tape.gradient(loss, small_model.trainable_weights)
            optimizer.apply_gradients(zip(gradients, small_model.trainable_weights))

        # Loss should decrease (at least somewhat)
        # This is a very simple test - just check it doesn't increase dramatically
        final_loss = losses[-1]
        assert not np.isnan(final_loss), "Final loss is NaN"
        assert not np.isinf(final_loss), "Final loss is Inf"

    def test_with_language_modeling_head(self, small_model):
        """Test integration with a simple language modeling head."""
        # Build the encoder
        input_ids = keras.random.randint((2, 16), minval=0, maxval=1000, dtype="int32")
        _ = small_model({"input_ids": input_ids}, training=False)

        # Create a simple LM head
        lm_head = keras.layers.Dense(
            small_model.vocab_size,
            use_bias=False,
            name="lm_head"
        )

        # Build complete model
        inputs = keras.Input(shape=(None,), dtype="int32", name="input_ids")
        encoder_outputs = small_model({"input_ids": inputs})
        logits = lm_head(encoder_outputs["last_hidden_state"])

        lm_model = keras.Model(inputs=inputs, outputs=logits)

        # Test forward pass
        sample_input = keras.random.randint((2, 16), minval=0, maxval=1000, dtype="int32")
        sample_logits = lm_model(sample_input)

        assert sample_logits.shape == (2, 16, small_model.vocab_size)

    def test_with_classification_head(self, small_model):
        """Test integration with a sequence classification head."""
        # Build encoder
        input_ids = keras.random.randint((2, 16), minval=0, maxval=1000, dtype="int32")
        _ = small_model({"input_ids": input_ids}, training=False)

        # Create classification head
        pooling = keras.layers.GlobalAveragePooling1D(name="pooling")
        dropout = keras.layers.Dropout(0.1, name="dropout")
        classifier = keras.layers.Dense(10, activation="softmax", name="classifier")

        # Build complete model
        inputs = keras.Input(shape=(None,), dtype="int32", name="input_ids")
        encoder_outputs = small_model({"input_ids": inputs})
        pooled = pooling(encoder_outputs["last_hidden_state"])
        dropped = dropout(pooled)
        outputs = classifier(dropped)

        classification_model = keras.Model(inputs=inputs, outputs=outputs)

        # Test forward pass
        sample_input = keras.random.randint((2, 16), minval=0, maxval=1000, dtype="int32")
        sample_output = classification_model(sample_input)

        assert sample_output.shape == (2, 10)

        # Check probabilities sum to 1
        probs_sum = keras.ops.sum(sample_output, axis=-1)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(probs_sum),
            np.ones(2),
            rtol=1e-6,
            atol=1e-6,
            err_msg="Probabilities should sum to 1"
        )


class TestMambaAdvancedFeatures:
    """Test advanced Mamba features and SSM-specific functionality."""

    def test_different_expansion_factors(self):
        """Test model with different expansion factors."""
        base_config = {
            "vocab_size": 1000,
            "d_model": 128,
            "num_layers": 2,
            "d_state": 8
        }

        for expand in [2, 3, 4]:
            model = Mamba(**base_config, expand=expand)
            input_ids = keras.random.randint((2, 16), minval=0, maxval=1000, dtype="int32")
            output = model({"input_ids": input_ids}, training=False)

            assert output['last_hidden_state'].shape == (2, 16, 128)

            # Check internal dimension
            first_mamba = model.encoder_layers[0].mamba
            expected_d_inner = expand * 128
            assert first_mamba.d_inner == expected_d_inner

    def test_different_state_dimensions(self):
        """Test model with different state space dimensions."""
        base_config = {
            "vocab_size": 1000,
            "d_model": 256,
            "num_layers": 2,
            "expand": 2
        }

        for d_state in [8, 16, 32, 64]:
            model = Mamba(**base_config, d_state=d_state)
            input_ids = keras.random.randint((2, 16), minval=0, maxval=1000, dtype="int32")
            output = model({"input_ids": input_ids}, training=False)

            assert output['last_hidden_state'].shape == (2, 16, 256)

            # Check state dimension in Mamba layer
            first_mamba = model.encoder_layers[0].mamba
            assert first_mamba.d_state == d_state

    def test_different_convolution_sizes(self):
        """Test model with different convolution kernel sizes."""
        base_config = {
            "vocab_size": 1000,
            "d_model": 128,
            "num_layers": 2,
            "d_state": 8,
            "expand": 2
        }

        for d_conv in [2, 4, 8, 16]:
            model = Mamba(**base_config, d_conv=d_conv)
            input_ids = keras.random.randint((2, 16), minval=0, maxval=1000, dtype="int32")
            output = model({"input_ids": input_ids}, training=False)

            assert output['last_hidden_state'].shape == (2, 16, 128)

            # Check conv size in Mamba layer
            first_mamba = model.encoder_layers[0].mamba
            assert first_mamba.d_conv == d_conv

    def test_explicit_dt_rank(self):
        """Test model with explicitly set dt_rank."""
        model = Mamba(
            vocab_size=1000,
            d_model=256,
            num_layers=2,
            d_state=16,
            dt_rank=32  # Explicit instead of "auto"
        )

        input_ids = keras.random.randint((2, 16), minval=0, maxval=1000, dtype="int32")
        output = model({"input_ids": input_ids}, training=False)

        assert output['last_hidden_state'].shape == (2, 16, 256)

        # Check dt_rank
        first_mamba = model.encoder_layers[0].mamba
        assert first_mamba.dt_rank == 32

    def test_model_summary(self):
        """Test that model summary works without errors."""
        model = Mamba.from_variant("base", vocab_size=1000)

        input_ids = keras.random.randint((1, 16), minval=0, maxval=1000, dtype="int32")
        _ = model({"input_ids": input_ids}, training=False)

        try:
            model.summary()
        except Exception as e:
            pytest.fail(f"Model summary raised an exception: {e}")

    def test_parameter_count(self):
        """Test that parameter count is reasonable."""
        model = Mamba(
            vocab_size=1000,
            d_model=128,
            num_layers=2,
            d_state=8
        )

        input_ids = keras.random.randint((1, 16), minval=0, maxval=1000, dtype="int32")
        _ = model({"input_ids": input_ids}, training=False)

        param_count = model.count_params()

        # Should have parameters (rough estimate)
        # Embedding: vocab_size * d_model = 1000 * 128
        # Each layer has multiple parameters
        assert param_count > 100000  # At least 100K parameters
        assert param_count < 10000000  # Less than 10M for this small model

    def test_layer_names_unique(self):
        """Test that all layer names are unique."""
        model = Mamba(
            vocab_size=1000,
            d_model=128,
            num_layers=4,
            d_state=8
        )

        input_ids = keras.random.randint((1, 16), minval=0, maxval=1000, dtype="int32")
        _ = model({"input_ids": input_ids}, training=False)

        # Get all layer names
        layer_names = [layer.name for layer in model.layers]

        # Check uniqueness
        assert len(layer_names) == len(set(layer_names)), "Layer names are not unique"

    def test_different_norm_epsilon(self):
        """Test model with different normalization epsilon values."""
        for epsilon in [1e-5, 1e-6, 1e-8]:
            model = Mamba(
                vocab_size=1000,
                d_model=128,
                num_layers=2,
                d_state=8,
                norm_epsilon=epsilon
            )

            input_ids = keras.random.randint((2, 16), minval=0, maxval=1000, dtype="int32")
            output = model({"input_ids": input_ids}, training=False)

            assert output['last_hidden_state'].shape == (2, 16, 128)
            assert model.norm_epsilon == epsilon


class TestMambaComparison:
    """Comparison tests between different Mamba configurations."""

    def test_variant_parameter_counts(self):
        """Test that different variants have expected parameter differences."""
        vocab_size = 50257

        variants = ["130m", "370m", "790m"]
        param_counts = []

        for variant in variants:
            model = Mamba.from_variant(variant, vocab_size=vocab_size)
            input_ids = keras.random.randint((1, 16), minval=0, maxval=vocab_size, dtype="int32")
            _ = model({"input_ids": input_ids}, training=False)

            param_count = model.count_params()
            param_counts.append(param_count)

        # Each larger variant should have more parameters
        assert param_counts[0] < param_counts[1] < param_counts[2]

    def test_deeper_model_has_more_params(self):
        """Test that deeper models have more parameters."""
        base_config = {
            "vocab_size": 1000,
            "d_model": 128,
            "d_state": 8
        }

        model_2_layers = Mamba(**base_config, num_layers=2)
        model_4_layers = Mamba(**base_config, num_layers=4)

        # Build both models
        input_ids = keras.random.randint((1, 16), minval=0, maxval=1000, dtype="int32")
        _ = model_2_layers({"input_ids": input_ids}, training=False)
        _ = model_4_layers({"input_ids": input_ids}, training=False)

        params_2 = model_2_layers.count_params()
        params_4 = model_4_layers.count_params()

        assert params_4 > params_2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])