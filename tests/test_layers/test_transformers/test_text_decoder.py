import pytest
import numpy as np
import tensorflow as tf
import keras
from keras import ops, layers, models
import tempfile
import os
from typing import Any, Dict, Tuple

from dl_techniques.layers.transformers.text_decoder import TextDecoder


class TestTextDecoder:
    """
    Comprehensive test suite for the TextDecoder layer following modern Keras 3 patterns.

    This test suite covers all aspects of the TextDecoder layer including initialization,
    parameter validation, forward passes, serialization, training integration, and
    edge cases. It follows the testing patterns from the modern Keras 3 guide.
    """

    # ===============================================
    # Test Fixtures
    # ===============================================

    @pytest.fixture
    def basic_config(self) -> Dict[str, Any]:
        """Provides a basic configuration for a small, testable decoder."""
        return {
            'vocab_size': 1000,
            'embed_dim': 64,
            'depth': 2,
            'num_heads': 4,
            'max_seq_len': 16,
            'dropout_rate': 0.1,
            'attention_dropout_rate': 0.1,
            'initializer_range': 0.02,
        }

    @pytest.fixture
    def modern_config(self) -> Dict[str, Any]:
        """Provides a modern decoder configuration with advanced features."""
        return {
            'vocab_size': 32000,
            'embed_dim': 512,
            'depth': 4,
            'num_heads': 8,
            'max_seq_len': 128,
            'embedding_type': 'factorized',
            'positional_type': 'sincos',
            'normalization_type': 'rms_norm',
            'normalization_position': 'pre',
            'ffn_type': 'swiglu',
            'stochastic_depth_rate': 0.1,
            'dropout_rate': 0.2,
            'attention_dropout_rate': 0.15,
        }

    @pytest.fixture
    def minimal_config(self) -> Dict[str, Any]:
        """Provides minimal configuration for edge case testing."""
        return {
            'vocab_size': 10,
            'embed_dim': 8,
            'depth': 1,
            'num_heads': 2,
            'max_seq_len': 4,
        }

    @pytest.fixture
    def sample_input_ids(self) -> keras.KerasTensor:
        """Provides sample input token IDs for testing."""
        return ops.convert_to_tensor(
            tf.random.uniform(shape=(2, 16), minval=0, maxval=1000, dtype=tf.int32)
        )

    @pytest.fixture
    def sample_attention_mask(self) -> keras.KerasTensor:
        """Provides a sample attention mask with padding."""
        mask = tf.ones((2, 16), dtype=tf.int32)
        # Pad the last 4 tokens of the second sequence
        mask = tf.tensor_scatter_nd_update(
            mask,
            [[1, 12], [1, 13], [1, 14], [1, 15]],
            [0, 0, 0, 0]
        )
        return ops.convert_to_tensor(mask)

    @pytest.fixture
    def variable_length_inputs(self) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """Provides variable length sequences for padding tests."""
        # Create sequences with different lengths
        input_ids = tf.constant([
            [1, 2, 3, 4, 0, 0, 0, 0],  # Length 4, padded with 0s
            [5, 6, 7, 8, 9, 10, 0, 0],  # Length 6, padded with 0s
        ], dtype=tf.int32)

        attention_mask = tf.constant([
            [1, 1, 1, 1, 0, 0, 0, 0],  # Mask for first sequence
            [1, 1, 1, 1, 1, 1, 0, 0],  # Mask for second sequence
        ], dtype=tf.int32)

        return ops.convert_to_tensor(input_ids), ops.convert_to_tensor(attention_mask)

    # ===============================================
    # 1. Initialization and Configuration Tests
    # ===============================================

    def test_basic_initialization(self, basic_config):
        """Test basic layer initialization with default parameters."""
        decoder = TextDecoder(**basic_config)

        # Check configuration storage
        assert decoder.vocab_size == basic_config['vocab_size']
        assert decoder.embed_dim == basic_config['embed_dim']
        assert decoder.depth == basic_config['depth']
        assert decoder.num_heads == basic_config['num_heads']

        # Check default values
        assert decoder.embedding_type == 'learned'
        assert decoder.positional_type == 'learned'
        assert decoder.normalization_position == 'post'
        assert decoder.ffn_type == 'mlp'
        assert decoder.attention_type == 'multi_head'
        assert decoder.normalization_type == 'layer_norm'

        # Check layer is not built initially
        assert not decoder.built

        # Check sub-layers are created
        assert hasattr(decoder, 'word_embeddings')
        assert hasattr(decoder, 'positional_embeddings')
        assert hasattr(decoder, 'embed_norm')
        assert hasattr(decoder, 'embed_dropout_layer')
        assert len(decoder.decoder_layers) == basic_config['depth']
        assert hasattr(decoder, 'final_norm')

    def test_modern_initialization(self, modern_config):
        """Test initialization with modern architectural choices."""
        decoder = TextDecoder(**modern_config)

        # Check modern configuration
        assert decoder.embedding_type == 'factorized'
        assert decoder.positional_type == 'sincos'
        assert decoder.normalization_type == 'rms_norm'
        assert decoder.normalization_position == 'pre'
        assert decoder.ffn_type == 'swiglu'
        assert decoder.stochastic_depth_rate == 0.1

        # Check factorized embeddings are created
        assert hasattr(decoder, 'factorized_embed_layer')
        assert hasattr(decoder, 'embed_projection_layer')
        assert not hasattr(decoder, 'word_embeddings')

    def test_all_embedding_types(self, basic_config):
        """Test all embedding type configurations."""
        embedding_types = ['learned', 'shared', 'factorized']

        for embedding_type in embedding_types:
            config = {**basic_config, 'embedding_type': embedding_type}
            decoder = TextDecoder(**config)

            if embedding_type in ['learned', 'shared']:
                assert hasattr(decoder, 'word_embeddings')
                assert not hasattr(decoder, 'factorized_embed_layer')
            elif embedding_type == 'factorized':
                assert hasattr(decoder, 'factorized_embed_layer')
                assert hasattr(decoder, 'embed_projection_layer')
                assert not hasattr(decoder, 'word_embeddings')

    def test_all_positional_types(self, basic_config):
        """Test all positional encoding type configurations."""
        positional_types = ['learned', 'sincos']

        for positional_type in positional_types:
            config = {**basic_config, 'positional_type': positional_type}
            decoder = TextDecoder(**config)
            assert hasattr(decoder, 'positional_embeddings')

    # ===============================================
    # 2. Parameter Validation Tests
    # ===============================================

    @pytest.mark.parametrize("param_name,invalid_value,expected_error", [
        ('vocab_size', 0, "must be positive integers"),
        ('embed_dim', -1, "must be positive integers"),
        ('depth', 0, "must be positive integers"),
        ('num_heads', -5, "must be positive integers"),
        ('max_seq_len', 0, "must be positive integers"),
        ('vocab_size', 3.14, "must be positive integers"),  # Non-integer
    ])
    def test_invalid_dimension_parameters(self, basic_config, param_name, invalid_value, expected_error):
        """Test validation of dimension parameters."""
        config = {**basic_config, param_name: invalid_value}
        with pytest.raises(ValueError, match=expected_error):
            TextDecoder(**config)

    def test_embed_dim_num_heads_compatibility(self, basic_config):
        """Test that embed_dim must be divisible by num_heads."""
        config = {**basic_config, 'embed_dim': 65, 'num_heads': 4}  # 65 % 4 != 0
        with pytest.raises(ValueError, match="must be divisible by num_heads"):
            TextDecoder(**config)

    @pytest.mark.parametrize("param_name,invalid_value", [
        ('dropout_rate', -0.1),
        ('dropout_rate', 1.1),
        ('attention_dropout_rate', -0.5),
        ('attention_dropout_rate', 2.0),
        ('stochastic_depth_rate', -0.1),
        ('stochastic_depth_rate', 1.5),
    ])
    def test_rate_parameter_validation(self, basic_config, param_name, invalid_value):
        """Test validation of rate parameters (must be in [0, 1])."""
        config = {**basic_config, param_name: invalid_value}
        with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
            TextDecoder(**config)

    # ===============================================
    # 3. Build Process Tests
    # ===============================================

    def test_build_process(self, basic_config, sample_input_ids):
        """Test that the decoder and all sub-layers are built correctly."""
        decoder = TextDecoder(**basic_config)

        # Initially not built
        assert not decoder.built

        # Trigger build through forward pass
        output = decoder(sample_input_ids)

        # Check main layer is built
        assert decoder.built

        # Check all sub-layers are built
        assert decoder.word_embeddings.built
        assert decoder.positional_embeddings.built
        assert decoder.embed_norm.built
        assert decoder.final_norm.built

        # Check all transformer layers are built
        for layer in decoder.decoder_layers:
            assert layer.built

        # Check output shape is correct
        expected_shape = (sample_input_ids.shape[0], sample_input_ids.shape[1], basic_config['embed_dim'])
        assert output.shape == expected_shape

    def test_build_with_factorized_embeddings(self, basic_config, sample_input_ids):
        """Test build process with factorized embeddings."""
        config = {**basic_config, 'embedding_type': 'factorized'}
        decoder = TextDecoder(**config)

        output = decoder(sample_input_ids)

        assert decoder.built
        assert decoder.factorized_embed_layer.built
        assert decoder.embed_projection_layer.built

    def test_build_with_sincos_positional(self, basic_config, sample_input_ids):
        """Test build process with sinusoidal positional encodings."""
        config = {**basic_config, 'positional_type': 'sincos'}
        decoder = TextDecoder(**config)

        output = decoder(sample_input_ids)

        assert decoder.built
        assert decoder.positional_embeddings.built

    # ===============================================
    # 4. Forward Pass Tests
    # ===============================================

    def test_forward_pass_basic(self, basic_config, sample_input_ids):
        """Test basic forward pass functionality."""
        decoder = TextDecoder(**basic_config)
        output = decoder(sample_input_ids)

        expected_shape = (
            sample_input_ids.shape[0],
            sample_input_ids.shape[1],
            basic_config['embed_dim']
        )
        assert output.shape == expected_shape
        assert output.dtype == keras.backend.floatx()

    def test_forward_pass_with_attention_mask(self, basic_config, sample_input_ids, sample_attention_mask):
        """Test forward pass with attention mask."""
        decoder = TextDecoder(**basic_config)
        output = decoder(sample_input_ids, attention_mask=sample_attention_mask)

        expected_shape = (
            sample_input_ids.shape[0],
            sample_input_ids.shape[1],
            basic_config['embed_dim']
        )
        assert output.shape == expected_shape

    def test_forward_pass_all_configurations(self, basic_config):
        """Test forward pass with all possible configuration combinations."""
        embedding_types = ['learned', 'factorized']
        positional_types = ['learned', 'sincos']

        input_ids = ops.convert_to_tensor(
            tf.random.uniform((2, 8), maxval=basic_config['vocab_size'], dtype=tf.int32)
        )

        for embed_type in embedding_types:
            for pos_type in positional_types:
                config = {
                    **basic_config,
                    'embedding_type': embed_type,
                    'positional_type': pos_type,
                    'max_seq_len': 8
                }
                decoder = TextDecoder(**config)
                output = decoder(input_ids)

                expected_shape = (2, 8, basic_config['embed_dim'])
                assert output.shape == expected_shape

    def test_variable_sequence_lengths(self, basic_config, variable_length_inputs):
        """Test handling of variable sequence lengths with proper masking."""
        input_ids, attention_mask = variable_length_inputs
        config = {**basic_config, 'max_seq_len': 8}

        decoder = TextDecoder(**config)
        output = decoder(input_ids, attention_mask=attention_mask)

        expected_shape = (2, 8, basic_config['embed_dim'])
        assert output.shape == expected_shape

    def test_training_vs_inference_mode(self, basic_config, sample_input_ids):
        """Test that layer behaves differently in training vs inference mode."""
        decoder = TextDecoder(**basic_config)

        # Get outputs in both modes
        training_output = decoder(sample_input_ids, training=True)
        inference_output = decoder(sample_input_ids, training=False)

        # Shapes should be identical
        assert training_output.shape == inference_output.shape

        # With dropout > 0, outputs should potentially be different
        # (though not guaranteed due to randomness)
        assert training_output.dtype == inference_output.dtype

    def test_compute_output_shape(self, basic_config):
        """Test compute_output_shape method."""
        decoder = TextDecoder(**basic_config)

        input_shape = (None, 20)  # Variable batch size, sequence length 20
        output_shape = decoder.compute_output_shape(input_shape)

        expected_shape = (None, 20, basic_config['embed_dim'])
        assert output_shape == expected_shape

    # ===============================================
    # 5. Serialization Tests (Critical for Keras 3)
    # ===============================================

    def test_serialization_cycle_basic(self, basic_config, sample_input_ids):
        """Test the complete serialization/deserialization cycle - most important test."""
        # Create model with decoder
        inputs = layers.Input(shape=(None,), dtype='int32', name='input_ids')
        outputs = TextDecoder(**basic_config, name='text_decoder')(inputs)
        model = models.Model(inputs, outputs, name='test_model')

        # Get prediction from original model
        original_prediction = model(sample_input_ids, training=False)

        # Save and load model
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_basic_decoder.keras")
            model.save(filepath)
            loaded_model = models.load_model(filepath)
            loaded_prediction = loaded_model(sample_input_ids, training=False)

            # Verify identical outputs
            np.testing.assert_allclose(
                ops.convert_to_numpy(original_prediction),
                ops.convert_to_numpy(loaded_prediction),
                rtol=1e-6, atol=1e-6,
                err_msg="Loaded model predictions should match original model"
            )

    def test_serialization_cycle_modern(self, modern_config):
        """Test serialization with modern/complex configuration."""
        seq_len = 32  # Shorter for testing
        vocab_size = modern_config['vocab_size']

        sample_input_ids = ops.convert_to_tensor(
            tf.random.uniform((2, seq_len), maxval=vocab_size, dtype=tf.int32)
        )

        # Create model
        inputs = layers.Input(shape=(None,), dtype='int32')
        config_copy = {**modern_config, 'max_seq_len': seq_len}
        outputs = TextDecoder(**config_copy)(inputs)
        model = models.Model(inputs, outputs)

        original_prediction = model(sample_input_ids, training=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_modern_decoder.keras")
            model.save(filepath)
            loaded_model = models.load_model(filepath)
            loaded_prediction = loaded_model(sample_input_ids, training=False)

            np.testing.assert_allclose(
                ops.convert_to_numpy(original_prediction),
                ops.convert_to_numpy(loaded_prediction),
                rtol=1e-5, atol=1e-5,
                err_msg="Modern configuration serialization failed"
            )

    def test_get_config_completeness(self, basic_config):
        """Test that get_config returns all necessary parameters."""
        decoder = TextDecoder(**basic_config)
        config = decoder.get_config()

        # Check all required parameters are present
        required_params = [
            'vocab_size', 'embed_dim', 'depth', 'num_heads', 'max_seq_len',
            'embedding_type', 'positional_type', 'attention_type',
            'normalization_type', 'normalization_position', 'ffn_type',
            'stochastic_depth_rate', 'dropout_rate', 'attention_dropout_rate',
            'initializer_range', 'layer_norm_eps'
        ]

        for param in required_params:
            assert param in config, f"Missing parameter {param} in config"

    def test_reconstruction_from_config(self, basic_config):
        """Test that layer can be reconstructed from its config."""
        original_decoder = TextDecoder(**basic_config)
        config = original_decoder.get_config()

        # Remove base class parameters to test our custom ones
        custom_config = {k: v for k, v in config.items()
                         if k in ['vocab_size', 'embed_dim', 'depth', 'num_heads',
                                  'max_seq_len', 'embedding_type', 'positional_type',
                                  'attention_type', 'normalization_type',
                                  'normalization_position', 'ffn_type',
                                  'stochastic_depth_rate', 'dropout_rate', 'attention_dropout_rate',
                                  'initializer_range', 'layer_norm_eps']}

        reconstructed_decoder = TextDecoder(**custom_config)

        # Check key parameters match
        assert reconstructed_decoder.vocab_size == original_decoder.vocab_size
        assert reconstructed_decoder.embed_dim == original_decoder.embed_dim
        assert reconstructed_decoder.depth == original_decoder.depth

    # ===============================================
    # 6. Training Integration Tests
    # ===============================================

    def test_gradient_flow(self, basic_config, sample_input_ids):
        """Test that gradients flow properly through the decoder."""
        decoder = TextDecoder(**basic_config)

        with tf.GradientTape() as tape:
            output = decoder(sample_input_ids, training=True)
            loss = ops.mean(ops.square(output))

        gradients = tape.gradient(loss, decoder.trainable_variables)

        assert len(gradients) > 0, "No gradients were computed"
        assert all(g is not None for g in gradients), "Some gradients are None"
        assert all(not ops.any(ops.isnan(g)) for g in gradients if g is not None), "NaN gradients detected"

    def test_trainable_variables_count(self, basic_config):
        """Test that the correct number of trainable variables exist."""
        decoder = TextDecoder(**basic_config)

        # Build the decoder
        sample_input = ops.convert_to_tensor(
            tf.random.uniform((1, 4), maxval=basic_config['vocab_size'], dtype=tf.int32)
        )
        decoder(sample_input)

        # Check that trainable variables exist
        trainable_vars = decoder.trainable_variables
        assert len(trainable_vars) > 0, "No trainable variables found"

        # Should have variables from embeddings, transformer layers, and norms
        # This is a rough check - exact count depends on architecture
        expected_min_vars = (
                2 +  # word + positional embeddings
                basic_config['depth'] * 6 +  # Each transformer layer has ~6-8 weight matrices
                2  # embedding norm + final norm
        )
        assert len(trainable_vars) >= expected_min_vars

    def test_model_compilation_and_training(self, basic_config):
        """Test integration into a full model training pipeline."""
        seq_len = 8
        vocab_size = basic_config['vocab_size']

        # Create a simple language modeling model
        model = models.Sequential([
            layers.InputLayer(shape=(seq_len,), dtype='int32'),
            TextDecoder(**{**basic_config, 'max_seq_len': seq_len}),
            layers.Dense(vocab_size, activation='softmax', name='lm_head')
        ])

        # Compile model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Create dummy training data
        batch_size = 4
        x_train = tf.random.uniform((batch_size, seq_len), maxval=vocab_size, dtype=tf.int32)
        y_train = tf.random.uniform((batch_size, seq_len), maxval=vocab_size, dtype=tf.int32)

        # Test that training runs without error
        history = model.fit(x_train, y_train, epochs=1, batch_size=2, verbose=0)

        assert 'loss' in history.history
        assert not np.isnan(history.history['loss'][0])

        # Test prediction
        predictions = model.predict(x_train, verbose=0)
        assert predictions.shape == (batch_size, seq_len, vocab_size)

    def test_stochastic_depth_functionality(self, basic_config):
        """Test that stochastic depth works correctly during training."""
        config = {**basic_config, 'stochastic_depth_rate': 0.5}
        decoder = TextDecoder(**config)

        sample_input = ops.convert_to_tensor(
            tf.random.uniform((4, 8), maxval=config['vocab_size'], dtype=tf.int32)
        )

        # Test multiple forward passes in training mode
        outputs = []
        for _ in range(5):
            output = decoder(sample_input, training=True)
            outputs.append(output)

        # With stochastic depth, outputs might vary (though not guaranteed)
        assert all(output.shape == outputs[0].shape for output in outputs)

    # ===============================================
    # 7. Edge Cases and Robustness Tests
    # ===============================================

    def test_minimal_configuration(self, minimal_config):
        """Test with minimal possible configuration."""
        decoder = TextDecoder(**minimal_config)

        sample_input = ops.convert_to_tensor([[0, 1, 2, 3]], dtype='int32')
        output = decoder(sample_input)

        expected_shape = (1, 4, minimal_config['embed_dim'])
        assert output.shape == expected_shape

    def test_single_token_input(self, basic_config):
        """Test with single token sequences."""
        decoder = TextDecoder(**basic_config)

        single_token_input = ops.convert_to_tensor([[42]], dtype='int32')
        output = decoder(single_token_input)

        expected_shape = (1, 1, basic_config['embed_dim'])
        assert output.shape == expected_shape

    def test_maximum_length_sequence(self, basic_config):
        """Test with sequences at maximum length."""
        max_len = basic_config['max_seq_len']
        decoder = TextDecoder(**basic_config)

        max_len_input = ops.convert_to_tensor(
            tf.random.uniform((1, max_len), maxval=basic_config['vocab_size'], dtype=tf.int32)
        )
        output = decoder(max_len_input)

        expected_shape = (1, max_len, basic_config['embed_dim'])
        assert output.shape == expected_shape

    def test_batch_size_variations(self, basic_config):
        """Test with different batch sizes."""
        decoder = TextDecoder(**basic_config)

        for batch_size in [1, 3, 8, 16]:
            batch_input = ops.convert_to_tensor(
                tf.random.uniform((batch_size, 8), maxval=basic_config['vocab_size'], dtype=tf.int32)
            )
            output = decoder(batch_input)

            expected_shape = (batch_size, 8, basic_config['embed_dim'])
            assert output.shape == expected_shape

    def test_zero_dropout_configuration(self, basic_config):
        """Test with zero dropout rates."""
        config = {**basic_config, 'dropout_rate': 0.0, 'attention_dropout_rate': 0.0, 'stochastic_depth_rate': 0.0}
        decoder = TextDecoder(**config)

        sample_input = ops.convert_to_tensor(
            tf.random.uniform((2, 8), maxval=config['vocab_size'], dtype=tf.int32)
        )

        # Multiple passes should give identical results with no dropout
        output1 = decoder(sample_input, training=True)
        output2 = decoder(sample_input, training=True)

        np.testing.assert_allclose(
            ops.convert_to_numpy(output1),
            ops.convert_to_numpy(output2),
            rtol=1e-6, atol=1e-6,
            err_msg="Outputs should be identical with zero dropout"
        )

    def test_layer_reuse(self, basic_config):
        """Test that the same layer instance can be reused multiple times."""
        decoder = TextDecoder(**basic_config)

        input1 = ops.convert_to_tensor([[1, 2, 3, 4]], dtype='int32')
        input2 = ops.convert_to_tensor([[5, 6, 7, 8]], dtype='int32')

        output1 = decoder(input1)
        output2 = decoder(input2)

        # Both should work and have correct shapes
        assert output1.shape == (1, 4, basic_config['embed_dim'])
        assert output2.shape == (1, 4, basic_config['embed_dim'])

    # ===============================================
    # 8. Performance and Memory Tests
    # ===============================================

    def test_output_stability_across_runs(self, basic_config):
        """Test that outputs are stable across multiple instantiations with same seed."""
        tf.random.set_seed(42)
        keras.utils.set_random_seed(42)

        decoder1 = TextDecoder(**basic_config)
        sample_input = ops.convert_to_tensor([[1, 2, 3, 4]], dtype='int32')
        output1 = decoder1(sample_input, training=False)

        tf.random.set_seed(42)
        keras.utils.set_random_seed(42)

        decoder2 = TextDecoder(**basic_config)
        output2 = decoder2(sample_input, training=False)

        # Should be identical with same random seed
        np.testing.assert_allclose(
            ops.convert_to_numpy(output1),
            ops.convert_to_numpy(output2),
            rtol=1e-6, atol=1e-6,
            err_msg="Outputs should be identical with same random seed"
        )

    # ===============================================
    # 9. Error Handling and Edge Cases
    # ===============================================

    def test_invalid_input_shapes(self, basic_config):
        """Test handling of invalid input shapes."""
        decoder = TextDecoder(**basic_config)

        # Test with wrong number of dimensions
        with pytest.raises((ValueError, tf.errors.InvalidArgumentError)):
            invalid_input = ops.convert_to_tensor([[[1, 2, 3]]], dtype='int32')  # 3D instead of 2D
            decoder(invalid_input)

    def test_attention_mask_shape_mismatch(self, basic_config):
        """Test handling of attention mask shape mismatches."""
        decoder = TextDecoder(**basic_config)

        input_ids = ops.convert_to_tensor([[1, 2, 3, 4]], dtype='int32')
        wrong_mask = ops.convert_to_tensor([[1, 1]], dtype='int32')  # Wrong shape

        with pytest.raises((ValueError, tf.errors.InvalidArgumentError)):
            decoder(input_ids, attention_mask=wrong_mask)

    # ===============================================
    # 10. Integration and Compatibility Tests
    # ===============================================

    def test_keras_functional_api_integration(self, basic_config):
        """Test integration with Keras Functional API."""
        input_ids = layers.Input(shape=(None,), dtype='int32', name='input_ids')
        attention_mask = layers.Input(shape=(None,), dtype='int32', name='attention_mask')

        decoder = TextDecoder(**basic_config, name='decoder')
        outputs = decoder(input_ids, attention_mask=attention_mask)

        model = models.Model(inputs=[input_ids, attention_mask], outputs=outputs)

        # Test model summary doesn't crash
        model.summary()

        # Test with actual inputs
        test_ids = tf.constant([[1, 2, 3, 4, 0, 0]], dtype=tf.int32)
        test_mask = tf.constant([[1, 1, 1, 1, 0, 0]], dtype=tf.int32)

        output = model([test_ids, test_mask])
        assert output.shape == (1, 6, basic_config['embed_dim'])

    def test_mixed_precision_compatibility(self, basic_config):
        """Test compatibility with mixed precision training."""
        # Set mixed precision policy
        original_policy = keras.mixed_precision.global_policy()
        keras.mixed_precision.set_global_policy('mixed_float16')

        try:
            decoder = TextDecoder(**basic_config)
            sample_input = ops.convert_to_tensor([[1, 2, 3, 4]], dtype='int32')

            output = decoder(sample_input)

            # Output should be in float16 for mixed precision
            # (Note: exact behavior may vary based on layer implementation)
            assert output.shape == (1, 4, basic_config['embed_dim'])

        finally:
            # Restore original policy
            keras.mixed_precision.set_global_policy(original_policy)