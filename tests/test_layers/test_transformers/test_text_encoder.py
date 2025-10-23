import pytest
import numpy as np
import tensorflow as tf
import keras
from keras import ops, layers, models
import tempfile
import os
from typing import Any, Dict

from dl_techniques.layers.transformers.text_encoder import (
    TextEncoder,
    create_text_encoder,
    create_bert_encoder,
    create_roberta_encoder,
    create_modern_encoder,
    create_efficient_encoder
)


# --- Test Class ---
class TestTextEncoder:
    """
    Comprehensive and modern test suite for the TextEncoder.
    This suite follows modern Keras 3 testing best practices and covers all
    architectural variations and factory patterns.
    """

    # --- Fixtures for Reusability ---
    @pytest.fixture
    def basic_config(self) -> Dict[str, Any]:
        """Provides a basic configuration for a small, testable encoder."""
        return {
            'vocab_size': 1000,
            'embed_dim': 64,
            'depth': 2,
            'num_heads': 4,
            'max_seq_len': 16,
        }

    @pytest.fixture
    def bert_config(self) -> Dict[str, Any]:
        """Provides BERT-style configuration."""
        return {
            'vocab_size': 30522,
            'embed_dim': 768,
            'depth': 12,
            'num_heads': 12,
            'max_seq_len': 512,
            'embedding_type': 'learned',
            'positional_type': 'learned',
            'use_token_type_embedding': True,
            'use_cls_token': True,
            'output_mode': 'cls'
        }

    @pytest.fixture
    def modern_config(self) -> Dict[str, Any]:
        """Provides modern encoder configuration with advanced features."""
        return {
            'vocab_size': 32000,
            'embed_dim': 512,
            'depth': 8,
            'num_heads': 8,
            'max_seq_len': 1024,
            'embedding_type': 'factorized',
            'positional_type': 'rope',
            'attention_type': 'differential',
            'normalization_type': 'rms_norm',
            'normalization_position': 'pre',
            'ffn_type': 'swiglu',
            'stochastic_depth_rate': 0.1,
        }

    @pytest.fixture
    def sample_input_ids(self) -> tf.Tensor:
        """Provides sample input token IDs for testing."""
        return tf.random.uniform(
            shape=(8, 16), minval=0, maxval=1000, dtype=tf.int32
        )

    @pytest.fixture
    def sample_token_type_ids(self) -> tf.Tensor:
        """Provides sample token type IDs for testing."""
        return tf.random.uniform(
            shape=(8, 16), minval=0, maxval=2, dtype=tf.int32
        )

    @pytest.fixture
    def sample_attention_mask(self) -> tf.Tensor:
        """Provides sample attention mask for testing."""
        # Create realistic attention mask with some padding
        mask = tf.ones((8, 16), dtype=tf.float32)
        # Simulate padding in second half of second sequence
        mask = tf.tensor_scatter_nd_update(
            mask, [[1, 8], [1, 9], [1, 10]], [0.0, 0.0, 0.0]
        )
        return mask

    # ===============================================
    # 1. Initialization and Build Tests
    # ===============================================
    def test_initialization_defaults(self, basic_config):
        """Tests encoder initialization with default parameters."""
        encoder = TextEncoder(**basic_config)
        assert not encoder.built
        assert encoder.embedding_type == 'learned'
        assert encoder.positional_type == 'learned'
        assert encoder.attention_type == 'multi_head'
        assert encoder.normalization_type == 'layer_norm'
        assert encoder.ffn_type == 'mlp'
        assert encoder.output_mode == 'none'
        assert not encoder.use_cls_token
        assert not encoder.use_token_type_embedding

    @pytest.mark.parametrize("embedding_type", ['learned', 'shared', 'factorized'])
    def test_initialization_embedding_types(self, basic_config, embedding_type):
        """Tests initialization with different embedding types."""
        config = {**basic_config, 'embedding_type': embedding_type}
        encoder = TextEncoder(**config)
        assert encoder.embedding_type == embedding_type

    @pytest.mark.parametrize("positional_type", ['learned', 'rope', 'dual_rope', 'sincos'])
    def test_initialization_positional_types(self, basic_config, positional_type):
        """Tests initialization with different positional encoding types."""
        config = {**basic_config, 'positional_type': positional_type}
        encoder = TextEncoder(**config)
        assert encoder.positional_type == positional_type

    @pytest.mark.parametrize("attention_type", [
        'multi_head', 'window', 'group_query', 'differential'
    ])
    def test_initialization_attention_types(self, basic_config, attention_type):
        """Tests initialization with different attention mechanisms."""
        config = {**basic_config, 'attention_type': attention_type}
        encoder = TextEncoder(**config)
        assert encoder.attention_type == attention_type

    @pytest.mark.parametrize("normalization_type", [
        'layer_norm', 'rms_norm', 'band_rms', 'dynamic_tanh'
    ])
    def test_initialization_normalization_types(self, basic_config, normalization_type):
        """Tests initialization with different normalization types."""
        config = {**basic_config, 'normalization_type': normalization_type}
        encoder = TextEncoder(**config)
        assert encoder.normalization_type == normalization_type

    @pytest.mark.parametrize("ffn_type", ['mlp', 'swiglu', 'differential', 'glu', 'geglu'])
    def test_initialization_ffn_types(self, basic_config, ffn_type):
        """Tests initialization with different FFN architectures."""
        config = {**basic_config, 'ffn_type': ffn_type}
        encoder = TextEncoder(**config)
        assert encoder.ffn_type == ffn_type

    def test_initialization_with_cls_token(self, basic_config):
        """Tests initialization with CLS token configuration."""
        config = {**basic_config, 'use_cls_token': True, 'output_mode': 'cls'}
        encoder = TextEncoder(**config)
        assert encoder.use_cls_token
        assert encoder.output_mode == 'cls'
        assert encoder.seq_len == basic_config['max_seq_len'] + 1

    def test_initialization_with_token_types(self, basic_config):
        """Tests initialization with token type embeddings."""
        config = {**basic_config, 'use_token_type_embedding': True, 'type_vocab_size': 3}
        encoder = TextEncoder(**config)
        assert encoder.use_token_type_embedding
        assert encoder.type_vocab_size == 3

    def test_build_process(self, basic_config, sample_input_ids):
        """Tests that encoder and all sub-layers are built correctly."""
        encoder = TextEncoder(**basic_config)
        assert not encoder.built
        output = encoder(sample_input_ids)
        assert encoder.built
        if encoder.word_embeddings:
            assert hasattr(encoder.word_embeddings, 'built')
            assert encoder.word_embeddings.built
        if encoder.factorized_embed_layer:
            assert hasattr(encoder.factorized_embed_layer, 'built')
            assert encoder.factorized_embed_layer.built


    def test_build_with_cls_token(self, basic_config, sample_input_ids):
        """Tests building with CLS token creates the token weight."""
        config = {**basic_config, 'use_cls_token': True}
        encoder = TextEncoder(**config)
        encoder(sample_input_ids)
        assert encoder.cls_token is not None
        assert encoder.cls_token.shape == (1, 1, basic_config['embed_dim'])

    # ===============================================
    # 2. Parameter Validation Tests
    # ===============================================
    def test_invalid_vocab_size(self):
        """Tests validation of vocab_size parameter."""
        with pytest.raises(ValueError, match="vocab_size must be positive"):
            TextEncoder(vocab_size=0, embed_dim=64)

    def test_invalid_embed_dim(self):
        """Tests validation of embed_dim parameter."""
        with pytest.raises(ValueError, match="embed_dim must be positive"):
            TextEncoder(vocab_size=1000, embed_dim=0)

    def test_embed_dim_num_heads_mismatch(self):
        """Tests validation of embed_dim and num_heads compatibility."""
        with pytest.raises(ValueError, match="embed_dim .* must be divisible by num_heads"):
            TextEncoder(vocab_size=1000, embed_dim=64, num_heads=5)

    def test_invalid_output_mode_cls_without_cls_token(self):
        """Tests validation of output_mode='cls' requiring use_cls_token=True."""
        with pytest.raises(ValueError, match="output_mode='cls' requires use_cls_token=True"):
            TextEncoder(vocab_size=1000, embed_dim=64, output_mode='cls', use_cls_token=False, num_heads=4)

    # ===============================================
    # 3. Forward Pass and Core Behavior Tests
    # ===============================================
    @pytest.mark.parametrize("output_mode", ['none', 'mean', 'max', 'first', 'last'])
    def test_forward_pass_output_modes(self, basic_config, sample_input_ids, output_mode):
        """Tests forward pass with different output modes."""
        config = {**basic_config, 'output_mode': output_mode}
        encoder = TextEncoder(**config)
        output = encoder(sample_input_ids, training=False)

        batch_size = sample_input_ids.shape[0]
        if output_mode == 'none':
            expected_shape = (batch_size, basic_config['max_seq_len'], basic_config['embed_dim'])
        else:
            expected_shape = (batch_size, basic_config['embed_dim'])

        assert output.shape == expected_shape
        assert not np.any(np.isnan(ops.convert_to_numpy(output)))

    def test_forward_pass_with_cls_token(self, basic_config, sample_input_ids):
        """Tests forward pass with CLS token."""
        config = {**basic_config, 'use_cls_token': True, 'output_mode': 'cls'}
        encoder = TextEncoder(**config)
        output = encoder(sample_input_ids, training=False)

        batch_size = sample_input_ids.shape[0]
        expected_shape = (batch_size, basic_config['embed_dim'])
        assert output.shape == expected_shape

    def test_forward_pass_with_token_type_ids(self, basic_config, sample_input_ids, sample_token_type_ids):
        """Tests forward pass with token type IDs."""
        config = {**basic_config, 'use_token_type_embedding': True}
        encoder = TextEncoder(**config)
        output = encoder(sample_input_ids, token_type_ids=sample_token_type_ids, training=False)

        batch_size = sample_input_ids.shape[0]
        expected_shape = (batch_size, basic_config['max_seq_len'], basic_config['embed_dim'])
        assert output.shape == expected_shape

    def test_forward_pass_with_attention_mask(self, basic_config, sample_input_ids, sample_attention_mask):
        """Tests forward pass with attention mask."""
        encoder = TextEncoder(**basic_config)
        output = encoder(sample_input_ids, attention_mask=sample_attention_mask, training=False)

        batch_size = sample_input_ids.shape[0]
        expected_shape = (batch_size, basic_config['max_seq_len'], basic_config['embed_dim'])
        assert output.shape == expected_shape

    def test_forward_pass_dict_inputs(self, basic_config, sample_input_ids, sample_token_type_ids):
        """Tests forward pass with dictionary inputs."""
        config = {**basic_config, 'use_token_type_embedding': True}
        encoder = TextEncoder(**config)

        inputs_dict = {
            'input_ids': sample_input_ids,
            'token_type_ids': sample_token_type_ids
        }
        output = encoder(inputs_dict, training=False)

        batch_size = sample_input_ids.shape[0]
        expected_shape = (batch_size, basic_config['max_seq_len'], basic_config['embed_dim'])
        assert output.shape == expected_shape

    def test_training_vs_inference_modes(self, basic_config, sample_input_ids):
        """Tests behavior difference between training and inference modes."""
        config = {**basic_config, 'dropout': 0.5, 'embed_dropout': 0.3}
        encoder = TextEncoder(**config)

        # Both should run without errors
        output_train = encoder(sample_input_ids, training=True)
        output_infer = encoder(sample_input_ids, training=False)

        assert output_train.shape == output_infer.shape
        assert not np.any(np.isnan(ops.convert_to_numpy(output_train)))
        assert not np.any(np.isnan(ops.convert_to_numpy(output_infer)))

    def test_get_sequence_features(self, basic_config, sample_input_ids):
        """Tests get_sequence_features method."""
        config = {**basic_config, 'output_mode': 'mean'}  # Different from 'none'
        encoder = TextEncoder(**config)

        # Should return full sequence regardless of output_mode
        sequence_features = encoder.get_sequence_features(inputs=sample_input_ids, training=False)

        batch_size = sample_input_ids.shape[0]
        expected_shape = (batch_size, basic_config['max_seq_len'], basic_config['embed_dim'])
        assert sequence_features.shape == expected_shape

    def test_get_pooled_features(self, basic_config, sample_input_ids):
        """Tests get_pooled_features method with different pooling modes."""
        encoder = TextEncoder(**basic_config)

        pooled_features = encoder.get_pooled_features(
            inputs=sample_input_ids, pooling_mode='mean', training=False
        )

        batch_size = sample_input_ids.shape[0]
        expected_shape = (batch_size, basic_config['embed_dim'])
        assert pooled_features.shape == expected_shape

    # ===============================================
    # 4. Serialization Tests (The Gold Standard)
    # ===============================================
    def test_full_serialization_cycle_basic(self, basic_config, sample_input_ids):
        """Tests full serialization cycle with basic configuration."""
        inputs = layers.Input(shape=sample_input_ids.shape[1:])
        outputs = TextEncoder(**basic_config)(inputs)
        model = models.Model(inputs, outputs)

        original_prediction = model(sample_input_ids, training=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_basic_encoder.keras")
            model.save(filepath)
            loaded_model = models.load_model(filepath)
            loaded_prediction = loaded_model(sample_input_ids, training=False)

            np.testing.assert_allclose(
                ops.convert_to_numpy(original_prediction),
                ops.convert_to_numpy(loaded_prediction),
                rtol=1e-6, atol=1e-6,
                err_msg="Basic encoder serialization failed"
            )

    def test_full_serialization_cycle_bert_style(self, bert_config):
        """Tests full serialization cycle with BERT-style configuration."""
        # Create inputs with the correct shape for this config
        batch_size = 2
        seq_len = bert_config['max_seq_len']
        input_ids = tf.random.uniform(
            (batch_size, seq_len), maxval=bert_config['vocab_size'], dtype=tf.int32
        )
        token_type_ids = tf.zeros_like(input_ids)

        input_ids_layer = layers.Input(shape=(seq_len,), name='input_ids')
        token_type_ids_layer = layers.Input(shape=(seq_len,), name='token_type_ids')

        encoder = TextEncoder(**bert_config)
        outputs = encoder({
            'input_ids': input_ids_layer,
            'token_type_ids': token_type_ids_layer
        })

        model = models.Model([input_ids_layer, token_type_ids_layer], outputs)
        original_prediction = model([input_ids, token_type_ids], training=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_bert_encoder.keras")
            model.save(filepath)
            loaded_model = models.load_model(filepath)
            loaded_prediction = loaded_model([input_ids, token_type_ids], training=False)

            np.testing.assert_allclose(
                ops.convert_to_numpy(original_prediction),
                ops.convert_to_numpy(loaded_prediction),
                rtol=1e-6, atol=1e-6,
                err_msg="BERT-style encoder serialization failed"
            )

    def test_full_serialization_cycle_modern(self, modern_config, sample_input_ids):
        """Tests full serialization cycle with modern encoder configuration."""
        # Adjust config to match test input size
        config = {**modern_config, 'max_seq_len': 16, 'vocab_size': 1000}

        inputs = layers.Input(shape=sample_input_ids.shape[1:])
        outputs = TextEncoder(**config)(inputs)
        model = models.Model(inputs, outputs)

        original_prediction = model(sample_input_ids, training=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_modern_encoder.keras")
            model.save(filepath)
            loaded_model = models.load_model(filepath)
            loaded_prediction = loaded_model(sample_input_ids, training=False)

            np.testing.assert_allclose(
                ops.convert_to_numpy(original_prediction),
                ops.convert_to_numpy(loaded_prediction),
                rtol=1e-6, atol=1e-6,
                err_msg="Modern encoder serialization failed"
            )

    # ===============================================
    # 5. Gradient and Training Integration Tests
    # ===============================================
    def test_gradient_flow(self, basic_config, sample_input_ids):
        """Tests gradient flow through encoder."""
        encoder = TextEncoder(**basic_config)
        x_var = tf.Variable(tf.cast(sample_input_ids, tf.float32))

        with tf.GradientTape() as tape:
            # Convert to int32 for embedding lookup
            x_int = tf.cast(x_var, tf.int32)
            output = encoder(x_int, training=True)
            loss = ops.mean(ops.square(output))

        gradients = tape.gradient(loss, encoder.trainable_variables)
        assert len(gradients) > 0, "No gradients were computed for encoder."
        assert all(g is not None for g in gradients), "A gradient is None in encoder."

    def test_model_training_loop_integration(self, basic_config):
        """Tests encoder integration in a standard training loop."""
        model = models.Sequential([
            layers.InputLayer(shape=(16,), dtype='int32'),
            TextEncoder(**basic_config),
            layers.GlobalAveragePooling1D(),
            layers.Dense(10)
        ])

        model.compile("adam", keras.losses.SparseCategoricalCrossentropy(from_logits=True))

        x_train = tf.random.uniform((32, 16), maxval=1000, dtype=tf.int32)
        y_train = tf.random.uniform([32], 0, 10, dtype=tf.int32)

        history = model.fit(x_train, y_train, epochs=1, batch_size=8, verbose=0)
        assert 'loss' in history.history
        assert not np.isnan(history.history['loss'][0]), "Loss became NaN during training."

    def test_stochastic_depth_behavior(self, basic_config, sample_input_ids):
        """Tests stochastic depth during training."""
        config = {**basic_config, 'stochastic_depth_rate': 0.2}
        encoder = TextEncoder(**config)

        # Multiple forward passes should potentially give different results due to stochastic depth
        outputs = []
        for _ in range(5):
            output = encoder(sample_input_ids, training=True)
            outputs.append(output)

        # All outputs should have same shape and no NaNs
        for output in outputs:
            assert output.shape == outputs[0].shape
            assert not np.any(np.isnan(ops.convert_to_numpy(output)))

    # ===============================================
    # 6. Factory Functions Tests
    # ===============================================
    def test_create_text_encoder_factory(self):
        """Tests the create_text_encoder factory function."""
        encoder = create_text_encoder(
            vocab_size=1000,
            embed_dim=64,
            depth=2,
            num_heads=4,
            max_seq_len=16
        )
        assert isinstance(encoder, TextEncoder)
        assert encoder.vocab_size == 1000
        assert encoder.embed_dim == 64

    def test_create_bert_encoder_factory(self):
        """Tests the create_bert_encoder factory function."""
        encoder = create_bert_encoder(vocab_size=1000, embed_dim=64, max_seq_len=16, num_heads=4)
        assert isinstance(encoder, TextEncoder)
        assert encoder.embedding_type == 'learned'
        assert encoder.positional_type == 'learned'
        assert encoder.use_token_type_embedding
        assert encoder.use_cls_token
        assert encoder.output_mode == 'cls'

    def test_create_roberta_encoder_factory(self):
        """Tests the create_roberta_encoder factory function."""
        encoder = create_roberta_encoder(vocab_size=1000, embed_dim=64, max_seq_len=16, num_heads=4)
        assert isinstance(encoder, TextEncoder)
        assert not encoder.use_token_type_embedding  # RoBERTa doesn't use token types
        assert encoder.use_cls_token
        assert encoder.output_mode == 'cls'

    def test_create_modern_encoder_factory(self):
        """Tests the create_modern_encoder factory function."""
        encoder = create_modern_encoder(vocab_size=1000, embed_dim=64, max_seq_len=16, num_heads=4)
        assert isinstance(encoder, TextEncoder)
        assert encoder.embedding_type == 'factorized'
        assert encoder.positional_type == 'rope'
        assert encoder.attention_type == 'differential'
        assert encoder.normalization_type == 'rms_norm'
        assert encoder.ffn_type == 'swiglu'

    def test_create_efficient_encoder_factory(self):
        """Tests the create_efficient_encoder factory function."""
        encoder = create_efficient_encoder(vocab_size=1000, embed_dim=64, max_seq_len=16, num_heads=4)
        assert isinstance(encoder, TextEncoder)
        assert encoder.embedding_type == 'factorized'
        assert encoder.stochastic_depth_rate > 0.0  # Should have stochastic depth

    def test_factory_parameter_validation(self):
        """Tests that factory functions validate parameters properly."""
        with pytest.raises(ValueError, match="vocab_size must be positive"):
            create_text_encoder(vocab_size=0, embed_dim=64)

        with pytest.raises(ValueError, match="embed_dim .* must be divisible by num_heads"):
            create_text_encoder(vocab_size=1000, embed_dim=64, num_heads=5)

    # ===============================================
    # 7. Configuration and Get Config Tests
    # ===============================================
    def test_get_config_completeness(self, basic_config):
        """Tests that get_config contains all initialization parameters."""
        encoder = TextEncoder(**basic_config)
        config = encoder.get_config()

        # Check all basic config parameters are present
        for key in basic_config:
            assert key in config, f"Missing {key} in get_config()"

        # Check additional parameters are present
        assert 'embedding_type' in config
        assert 'positional_type' in config
        assert 'attention_type' in config
        assert 'output_mode' in config

    def test_config_reconstruction(self, basic_config):
        """Tests that encoder can be reconstructed from config."""
        original_encoder = TextEncoder(**basic_config)
        config = original_encoder.get_config()
        reconstructed_encoder = TextEncoder.from_config(config)

        # Key attributes should match
        assert reconstructed_encoder.vocab_size == original_encoder.vocab_size
        assert reconstructed_encoder.embed_dim == original_encoder.embed_dim
        assert reconstructed_encoder.depth == original_encoder.depth
        assert reconstructed_encoder.embedding_type == original_encoder.embedding_type

    def test_compute_output_shape(self, basic_config):
        """Tests compute_output_shape method."""
        encoder = TextEncoder(**basic_config)

        input_shape = (None, 16)  # (batch_size, seq_len)
        output_shape = encoder.compute_output_shape(input_shape)

        expected_shape = (None, 16, 64)  # (batch_size, seq_len, embed_dim) for 'none' mode
        assert output_shape == expected_shape

    def test_compute_output_shape_pooled(self, basic_config):
        """Tests compute_output_shape with pooled output modes."""
        config = {**basic_config, 'output_mode': 'mean'}
        encoder = TextEncoder(**config)

        input_shape = (None, 16)
        output_shape = encoder.compute_output_shape(input_shape)

        expected_shape = (None, 64)  # (batch_size, embed_dim)
        assert output_shape == expected_shape

    # ===============================================
    # 8. Advanced Architecture Tests
    # ===============================================
    @pytest.mark.parametrize("normalization_position", ['pre', 'post'])
    def test_normalization_positions(self, basic_config, sample_input_ids, normalization_position):
        """Tests different normalization positions."""
        config = {**basic_config, 'normalization_position': normalization_position}
        encoder = TextEncoder(**config)
        output = encoder(sample_input_ids, training=False)

        expected_shape = (sample_input_ids.shape[0], basic_config['max_seq_len'], basic_config['embed_dim'])
        assert output.shape == expected_shape

    def test_rope_with_custom_parameters(self, basic_config, sample_input_ids):
        """Tests RoPE with custom theta and percentage parameters."""
        config = {
            **basic_config,
            'positional_type': 'rope',
            'rope_theta': 100000.0,
            'rope_percentage': 0.5
        }
        encoder = TextEncoder(**config)
        output = encoder(sample_input_ids, training=False)

        expected_shape = (sample_input_ids.shape[0], basic_config['max_seq_len'], basic_config['embed_dim'])
        assert output.shape == expected_shape

    def test_factorized_embeddings_with_custom_dim(self, basic_config, sample_input_ids):
        """Tests factorized embeddings with custom factorization dimension."""
        config = {
            **basic_config,
            'embedding_type': 'factorized',
            'embedding_args': {'factorized_dim': 32}
        }
        encoder = TextEncoder(**config)
        output = encoder(sample_input_ids, training=False)

        expected_shape = (sample_input_ids.shape[0], basic_config['max_seq_len'], basic_config['embed_dim'])
        assert output.shape == expected_shape

    def test_custom_layer_args(self, basic_config, sample_input_ids):
        """Tests encoder with custom arguments for sub-components."""
        config = {
            **basic_config,
            'attention_args': {'use_bias': False},
            'norm_args': {'epsilon': 1e-6},
            'ffn_args': {'use_bias': False}
        }
        encoder = TextEncoder(**config)
        output = encoder(sample_input_ids, training=False)

        expected_shape = (sample_input_ids.shape[0], basic_config['max_seq_len'], basic_config['embed_dim'])
        assert output.shape == expected_shape

    # ===============================================
    # 9. Edge Cases and Error Handling
    # ===============================================
    def test_empty_sequence_handling(self, basic_config):
        """Tests behavior with minimum sequence length."""
        config = {**basic_config, 'max_seq_len': 1}
        encoder = TextEncoder(**config)

        input_ids = tf.random.uniform((2, 1), maxval=1000, dtype=tf.int32)
        output = encoder(input_ids, training=False)

        expected_shape = (2, 1, basic_config['embed_dim'])
        assert output.shape == expected_shape

    def test_large_vocabulary_handling(self, basic_config, sample_input_ids):
        """Tests encoder with large vocabulary size."""
        config = {**basic_config, 'vocab_size': 100000}
        encoder = TextEncoder(**config)
        output = encoder(sample_input_ids, training=False)

        expected_shape = (sample_input_ids.shape[0], basic_config['max_seq_len'], basic_config['embed_dim'])
        assert output.shape == expected_shape

    def test_attention_mask_with_cls_token(self, basic_config, sample_input_ids, sample_attention_mask):
        """Tests attention mask behavior when CLS token is added."""
        config = {**basic_config, 'use_cls_token': True, 'output_mode': 'cls'}
        encoder = TextEncoder(**config)

        output = encoder(sample_input_ids, attention_mask=sample_attention_mask, training=False)

        expected_shape = (sample_input_ids.shape[0], basic_config['embed_dim'])
        assert output.shape == expected_shape

    def test_mixed_precision_compatibility(self, basic_config, sample_input_ids):
        """Tests encoder compatibility with mixed precision training."""
        # Enable mixed precision
        policy = keras.mixed_precision.Policy('mixed_float16')
        keras.mixed_precision.set_global_policy(policy)

        try:
            encoder = TextEncoder(**basic_config)
            output = encoder(sample_input_ids, training=False)

            # Output should be float16 for mixed precision
            assert output.dtype == tf.float16

        finally:
            # Reset policy
            keras.mixed_precision.set_global_policy('float32')

    # ===============================================
    # 10. Performance and Memory Tests
    # ===============================================
    def test_multiple_sequence_lengths(self, basic_config):
        """Tests encoder with various sequence lengths within max_seq_len."""
        encoder = TextEncoder(**basic_config)

        for seq_len in [4, 8, 12, 16]:
            input_ids = tf.random.uniform((2, seq_len), maxval=1000, dtype=tf.int32)
            output = encoder(input_ids, training=False)

            expected_shape = (2, seq_len, basic_config['embed_dim'])
            assert output.shape == expected_shape

    def test_batch_size_variations(self, basic_config, sample_input_ids):
        """Tests encoder with different batch sizes."""
        encoder = TextEncoder(**basic_config)

        for batch_size in [1, 4, 8]:
            input_ids = sample_input_ids[:batch_size]
            output = encoder(input_ids, training=False)

            expected_shape = (batch_size, basic_config['max_seq_len'], basic_config['embed_dim'])
            assert output.shape == expected_shape

    def test_memory_efficiency_with_gradient_checkpointing(self, basic_config, sample_input_ids):
        """Tests memory-efficient training configurations."""
        config = {
            **basic_config,
            'depth': 4,  # Deeper for testing
            'stochastic_depth_rate': 0.1,
            'dropout': 0.1,
        }
        encoder = TextEncoder(**config)

        # Should work with gradient tape for training
        with tf.GradientTape() as tape:
            output = encoder(sample_input_ids, training=True)
            loss = ops.mean(ops.square(output))

        gradients = tape.gradient(loss, encoder.trainable_variables)
        assert all(g is not None for g in gradients)

    # ===============================================
    # 11. Backwards Compatibility Tests
    # ===============================================
    def test_serialization_backwards_compatibility(self, basic_config, sample_input_ids):
        """Tests that saved models maintain compatibility across versions."""
        # This test ensures that the get_config/from_config cycle preserves all functionality
        encoder = TextEncoder(**basic_config)
        encoder.build(sample_input_ids.shape)  # Build the original encoder

        # Simulate save/load cycle
        config = encoder.get_config()
        reconstructed = TextEncoder.from_config(config)

        # Re-build the reconstructed layer
        reconstructed.build(sample_input_ids.shape)
        # Set weights to be identical
        reconstructed.set_weights(encoder.get_weights())

        # Both should produce the same architecture
        original_output = encoder(sample_input_ids, training=False)
        reconstructed_output = reconstructed(sample_input_ids, training=False)

        assert original_output.shape == reconstructed_output.shape
        np.testing.assert_allclose(
            ops.convert_to_numpy(original_output),
            ops.convert_to_numpy(reconstructed_output),
            rtol=1e-6, atol=1e-6
        )