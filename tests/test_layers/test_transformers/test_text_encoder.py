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
            # D-001: 'rope' is not wired into TextEncoder (fail-loud); modern
            # encoder uses 'learned' positional embeddings.
            'positional_type': 'learned',
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

    @pytest.mark.parametrize("positional_type", ['learned', 'sincos'])
    def test_initialization_positional_types(self, basic_config, positional_type):
        """Tests initialization with the supported positional encoding types."""
        config = {**basic_config, 'positional_type': positional_type}
        encoder = TextEncoder(**config)
        assert encoder.positional_type == positional_type

    @pytest.mark.parametrize("positional_type", ['rope', 'dual_rope'])
    def test_rope_positional_types_fail_loud(self, basic_config, positional_type):
        """D-001: rope/dual_rope are not wired in and must fail loud, not silently
        produce a positionless model."""
        config = {**basic_config, 'positional_type': positional_type}
        with pytest.raises(NotImplementedError):
            TextEncoder(**config)

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
        config = {**basic_config, 'dropout_rate': 0.5, 'embed_dropout_rate': 0.3}
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
        assert encoder.positional_type == 'learned'  # D-001: repointed from 'rope'
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
        """D-001: RoPE is not wired into TextEncoder; requesting it must fail loud
        regardless of the custom rope_theta/rope_percentage values."""
        config = {
            **basic_config,
            'positional_type': 'rope',
            'rope_theta': 100000.0,
            'rope_percentage': 0.5
        }
        with pytest.raises(NotImplementedError):
            TextEncoder(**config)

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
            'dropout_rate': 0.1,
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

# =====================================================================
# G-01 (finding F-24) -- coverage over the three formerly-BROKEN
# `SequencePooling` strategies.
#
# `TextEncoder.output_mode` is the full, unrestricted `PoolingStrategy`
# surface, and `call()` forwards the caller's `attention_mask` straight into
# `self.pooling_layer(x, mask=..., ...)`. Until this section existed the module
# parametrized `output_mode` over `['none','mean','max','first','last']` ONLY,
# so `weighted`, `top_k_mean` and `top_k_max` had NEVER been exercised here --
# masked or unmasked.
#
# Those three strategies leaked a masked position into the pooled output
# (F-24): `WeightedPooling` multiplied the position weights by the mask BEFORE
# the softmax, so a masked position kept weight `softmax(0) != 0`; the
# `top_k_*` branches ranked the MASKED norms but gathered from the UNMASKED
# inputs, which leaks whenever `k` exceeds a row's kept count. The root fix
# lives in `layers/sequence_pooling/`; `VisionEncoder`'s `NotImplementedError`
# containment (prior-plan D-013) is gone, and `TextEncoder` -- which never had
# any containment -- gets the coverage instead.
#
# BOUNDARY DISCIPLINE (measured, not assumed): the leak is exactly 0.0 even
# BEFORE the fix when `k == kept_count`, so a probe resting on that cell proves
# nothing. `TextEncoder` never exposes `top_k`, so `k = min(10, seq_len)` with
# `SequencePooling`'s default of 10. The geometry below is chosen so that
# `k = 10 < seq_len` (the ranking sentinel is observable -- with `k >= seq_len`
# every position is selected and that half of the fix is a no-op) AND
# `k > kept_count` (the gathered-validity half is observable).
# `test_the_probe_geometry_is_not_on_the_k_equals_kept_count_boundary` pins
# both inequalities so the suite cannot silently drift onto the dead cell.
# =====================================================================

G01_STRATEGIES = ['weighted', 'top_k_mean', 'top_k_max']

G01_CFG: Dict[str, Any] = {
    'vocab_size': 97,
    'embed_dim': 32,
    'depth': 2,
    'num_heads': 4,
    'max_seq_len': 16,
    'use_bias': True,
}
G01_B = 2
G01_S = 12                      # > 10 so `k = min(10, seq_len) = 10 < seq_len`
G01_MASKED_POSITIONS = [8, 9, 10, 11]   # kept = 8 (no CLS) / 9 (CLS) < k = 10
G01_LIVE_POS = 0                # perturbed for the mandatory live control


def _g01_encoder(seed: int = 1234, **overrides: Any) -> TextEncoder:
    """Build a `TextEncoder` and assign EVERY weight from a seeded RNG.

    Fresh Keras initialisers leave biases at zero, which makes a masking site
    unobservable by construction (invariant I8 / prior-plan D-008): a zeroed
    bias can null the very activation whose leakage the probe is trying to
    measure. These fixtures therefore put the encoder in the state a *trained*
    model is in -- non-zero biases, non-unit normalisation gains.

    :param seed: RNG seed for the weight assignment.
    :param overrides: Keyword overrides merged over ``G01_CFG``.
    :return: A built encoder with fully randomised, non-degenerate weights.
    """
    cfg = {**G01_CFG, **overrides}
    encoder = TextEncoder(**cfg)
    encoder.build((None, G01_S))

    rng = np.random.default_rng(seed)
    saw_nonzero_bias = False
    for w in encoder.weights:
        shape = tuple(w.shape)
        name = w.path.split('/')[-1]
        if 'gamma' in name or 'scale' in name:
            # Keep normalisation gains near 1 -- a near-zero gain collapses the
            # signal and makes the whole probe vacuous.
            value = 1.0 + 0.05 * rng.normal(size=shape)
        elif 'beta' in name or 'bias' in name:
            value = 0.05 * rng.normal(size=shape)
            saw_nonzero_bias = True
        else:
            value = 0.1 * rng.normal(size=shape)
        w.assign(ops.cast(ops.convert_to_tensor(value), w.dtype))

    assert saw_nonzero_bias, (
        "Fixture is degenerate: no bias weight was assigned a non-zero value, "
        "so masking sites downstream of a zeroed activation cannot be observed."
    )
    return encoder


def _g01_ids(seed: int = 7) -> np.ndarray:
    """Return a deterministic ``(B, S)`` batch of token ids.

    Ids start at 1: the word embedding is built with ``mask_zero=True``, so id
    0 would attach a SECOND, implicit Keras mask and confound the explicit
    ``attention_mask`` this suite is measuring.
    """
    rng = np.random.default_rng(seed)
    return rng.integers(1, G01_CFG['vocab_size'], size=(G01_B, G01_S)).astype('int32')


def _g01_perturb(
        input_ids: np.ndarray,
        positions: list,
        rows: Any = None,
        seed: int = 99,
) -> np.ndarray:
    """Replace the token id at ``positions`` with a DIFFERENT random id.

    The replacement is guaranteed to differ from the original, so the
    perturbation cannot silently be a no-op (which would make both the live
    control and the isolation assertion meaningless).

    :param input_ids: Batch of token ids ``(B, S)``.
    :param positions: Sequence positions to perturb.
    :param rows: Batch rows to perturb; ``None`` means all rows.
    :param seed: RNG seed for the replacement ids.
    :return: A copy of ``input_ids`` with those cells changed.
    """
    rng = np.random.default_rng(seed)
    out = np.array(input_ids, copy=True)
    target_rows = range(out.shape[0]) if rows is None else rows
    for row in target_rows:
        for pos in positions:
            new = int(rng.integers(1, G01_CFG['vocab_size']))
            while new == out[row, pos]:
                new = int(rng.integers(1, G01_CFG['vocab_size']))
            out[row, pos] = new
    return out


def _g01_mask(masked_per_row: list) -> np.ndarray:
    """Return a ``(B, S)`` keep-mask (1 = attend) with the listed positions masked."""
    m = np.ones((G01_B, G01_S), dtype='float32')
    for row, positions in enumerate(masked_per_row):
        for pos in positions:
            m[row, pos] = 0.0
    return m


def _g01_np(x: Any) -> np.ndarray:
    return ops.convert_to_numpy(x)


def _g01_pooling_mask(encoder: TextEncoder, mask: np.ndarray) -> Any:
    """Reproduce `TextEncoder.call`'s CLS extension of the pooling mask."""
    m = ops.convert_to_tensor(mask)
    if encoder.use_cls_token:
        m = ops.concatenate([ops.ones((mask.shape[0], 1), dtype=m.dtype), m], axis=1)
    return m


class TestFormerlyBrokenPoolingStrategies:
    """G-01: `weighted` / `top_k_mean` / `top_k_max` through `TextEncoder`.

    Every masked test carries a LIVE CONTROL -- the same perturbation applied
    to an UNMASKED position must move the pooled output by a wide margin -- so
    a test that passes because *nothing* moves is impossible.
    """

    @pytest.mark.parametrize("use_cls_token", [False, True])
    def test_the_probe_geometry_is_not_on_the_k_equals_kept_count_boundary(
            self, use_cls_token
    ):
        """Pin the two inequalities the whole masked suite below depends on.

        `TextEncoder` does not expose `top_k`, so the effective `k` is
        `min(SequencePooling.top_k, seq_len)`. Both halves of the F-24 fix are
        unobservable unless:

        * `k < seq_len`   -- otherwise every position is selected and the
          masked-norm RANKING sentinel is a no-op;
        * `k > kept_count` -- the leak measures exactly 0.0 at
          `k == kept_count` even BEFORE the fix, so a probe on that cell would
          be vacuous.

        If `SequencePooling`'s default `top_k` ever moves, this fires first and
        names the reason rather than letting the suite quietly go green on a
        dead cell.
        """
        encoder = _g01_encoder(output_mode='top_k_mean', use_cls_token=use_cls_token)
        assert encoder.pooling_layer.top_k == 10, (
            "SequencePooling's default top_k moved; re-derive G01_S and "
            "G01_MASKED_POSITIONS so that k < seq_len and k > kept_count."
        )

        seq_len = G01_S + (1 if use_cls_token else 0)
        k = min(encoder.pooling_layer.top_k, seq_len)
        kept = seq_len - len(G01_MASKED_POSITIONS)

        assert k < seq_len, f"k={k} must be < seq_len={seq_len}"
        assert k > kept, f"k={k} must be > kept_count={kept}"

    @pytest.mark.parametrize("use_cls_token", [False, True])
    @pytest.mark.parametrize("output_mode", G01_STRATEGIES)
    def test_masked_position_is_isolated_through_call(self, output_mode, use_cls_token):
        """SC-4: perturbing MASKED tokens must leave `call()`'s output bit-identical.

        ALL FOUR masked positions are perturbed, not just one, and that is
        load-bearing rather than thorough-for-its-own-sake. `ops.top_k` breaks
        the tie between equally-ranked masked positions by ASCENDING INDEX, so
        with `k - kept_count == 2` only the two LOWEST-indexed masked positions
        are ever selected. A probe that perturbed only the last masked position
        measured 0.0 even with the gathered-validity half of the F-24 fix
        neutered -- i.e. it was half-vacuous. Perturbing the whole masked set
        removes the dependence on `top_k`'s tie-breaking order.
        """
        encoder = _g01_encoder(output_mode=output_mode, use_cls_token=use_cls_token)
        input_ids = _g01_ids()
        mask = ops.convert_to_tensor(_g01_mask([G01_MASKED_POSITIONS] * G01_B))

        base = _g01_np(encoder(
            ops.convert_to_tensor(input_ids), attention_mask=mask, training=False
        ))

        live = _g01_np(encoder(
            ops.convert_to_tensor(_g01_perturb(input_ids, [G01_LIVE_POS])),
            attention_mask=mask, training=False,
        ))
        live_delta = float(np.max(np.abs(live - base)))
        assert live_delta > 1e-2, (
            f"Vacuous probe for output_mode={output_mode!r}, use_cls_token="
            f"{use_cls_token}: perturbing the UNMASKED position "
            f"{G01_LIVE_POS} moved the output by only {live_delta:.6e}, so the "
            f"isolation assertion below proves nothing."
        )

        leaked = _g01_np(encoder(
            ops.convert_to_tensor(_g01_perturb(input_ids, G01_MASKED_POSITIONS)),
            attention_mask=mask, training=False,
        ))
        np.testing.assert_allclose(
            leaked, base, rtol=0, atol=0,
            err_msg=(
                f"output_mode={output_mode!r}, use_cls_token={use_cls_token}: a "
                f"MASKED token leaked into the pooled output by "
                f"{float(np.max(np.abs(leaked - base))):.6e}; required 0.0."
            ),
        )

    @pytest.mark.parametrize("use_cls_token", [False, True])
    @pytest.mark.parametrize("output_mode", G01_STRATEGIES)
    def test_isolation_holds_at_the_pooling_layer_itself(self, output_mode, use_cls_token):
        """The same contract measured AT the pooling layer, not only through `call()`.

        Kept distinct from the `call()`-level test on purpose: it proves the
        isolation comes from the POOLING fix and not merely from self-attention
        already excluding the masked position upstream. `get_sequence_features`
        returns the un-pooled sequence (CLS included when configured), which is
        exactly the tensor `call()` hands to `self.pooling_layer`.

        Perturbs the WHOLE masked set, for the `top_k` tie-order reason spelled
        out on `test_masked_position_is_isolated_through_call`.
        """
        encoder = _g01_encoder(output_mode=output_mode, use_cls_token=use_cls_token)
        input_ids = _g01_ids()
        mask_np = _g01_mask([G01_MASKED_POSITIONS] * G01_B)
        mask = ops.convert_to_tensor(mask_np)
        pooling_mask = _g01_pooling_mask(encoder, mask_np)

        def _pooled(arr: np.ndarray) -> np.ndarray:
            features = encoder.get_sequence_features(
                inputs=ops.convert_to_tensor(arr),
                attention_mask=mask,
                training=False,
            )
            return _g01_np(encoder.pooling_layer(
                features, mask=pooling_mask, training=False
            ))

        base = _pooled(input_ids)
        live = _pooled(_g01_perturb(input_ids, [G01_LIVE_POS]))
        live_delta = float(np.max(np.abs(live - base)))
        assert live_delta > 1e-2, (
            f"Vacuous probe for output_mode={output_mode!r}: the live control "
            f"moved only {live_delta:.6e} at the pooling layer."
        )

        leaked = _pooled(_g01_perturb(input_ids, G01_MASKED_POSITIONS))
        delta = float(np.max(np.abs(leaked - base)))
        assert delta == 0.0, (
            f"output_mode={output_mode!r}, use_cls_token={use_cls_token}: the "
            f"masked position leaked into the pooled output by {delta:.6e}; "
            f"F-24 has regressed in `layers/sequence_pooling/`."
        )

    @pytest.mark.parametrize("output_mode", G01_STRATEGIES)
    def test_heterogeneous_kept_counts_are_isolated(self, output_mode):
        """`k` is batch-GLOBAL while the kept count is PER-ROW -- that asymmetry IS the defect.

        Row 0 masks a single position (kept 11 > k = 10); row 1 masks five
        (kept 7 < k = 10). A fix that clamped `k` to the batch-wide minimum
        would make row 0's answer depend on row 1's mask; a fix that does not
        exclude invalid SELECTED positions leaks in row 1.
        """
        encoder = _g01_encoder(output_mode=output_mode)
        input_ids = _g01_ids(seed=11)
        masked_per_row = [[11], [7, 8, 9, 10, 11]]
        mask = ops.convert_to_tensor(_g01_mask(masked_per_row))

        base = _g01_np(encoder(
            ops.convert_to_tensor(input_ids), attention_mask=mask, training=False
        ))

        live = _g01_np(encoder(
            ops.convert_to_tensor(_g01_perturb(input_ids, [G01_LIVE_POS])),
            attention_mask=mask, training=False,
        ))
        live_delta = float(np.max(np.abs(live - base)))
        assert live_delta > 1e-2, (
            f"Vacuous probe for output_mode={output_mode!r}: the live control "
            f"moved only {live_delta:.6e} with heterogeneous kept counts."
        )

        perturbed = np.array(input_ids, copy=True)
        for row, positions in enumerate(masked_per_row):
            perturbed = _g01_perturb(perturbed, positions, rows=[row])
        leaked = _g01_np(encoder(
            ops.convert_to_tensor(perturbed), attention_mask=mask, training=False
        ))
        np.testing.assert_allclose(
            leaked, base, rtol=0, atol=0,
            err_msg=(
                f"output_mode={output_mode!r}: masked positions leaked with "
                f"heterogeneous per-row kept counts (11 and 7) by "
                f"{float(np.max(np.abs(leaked - base))):.6e}; required 0.0."
            ),
        )

    @pytest.mark.parametrize("use_cls_token", [False, True])
    @pytest.mark.parametrize("output_mode", G01_STRATEGIES)
    def test_unmasked_forward_pass(self, output_mode, use_cls_token):
        """`mask=None` coverage the module also lacked: shape, finiteness, mask liveness.

        The final assertion is the non-vacuity control for the whole unmasked
        family: supplying a mask must CHANGE the answer. If it did not, every
        "isolation" result above would be explained by the mask being ignored
        outright rather than honoured.
        """
        encoder = _g01_encoder(output_mode=output_mode, use_cls_token=use_cls_token)
        input_ids = ops.convert_to_tensor(_g01_ids())

        out = encoder(input_ids, training=False)
        expected = encoder.compute_output_shape((G01_B, G01_S))
        assert tuple(out.shape) == (G01_B, G01_CFG['embed_dim'])
        assert tuple(out.shape)[1:] == tuple(expected)[1:]
        assert np.all(np.isfinite(_g01_np(out)))

        explicit_none = encoder(input_ids, attention_mask=None, training=False)
        np.testing.assert_allclose(
            _g01_np(explicit_none), _g01_np(out), rtol=0, atol=0,
            err_msg="An explicit attention_mask=None must equal the default path.",
        )

        masked = encoder(
            input_ids,
            attention_mask=ops.convert_to_tensor(
                _g01_mask([G01_MASKED_POSITIONS] * G01_B)
            ),
            training=False,
        )
        assert float(np.max(np.abs(_g01_np(masked) - _g01_np(out)))) > 1e-3, (
            f"output_mode={output_mode!r}: masking four positions did not change "
            f"the output at all, so the mask is being ignored."
        )

    @pytest.mark.parametrize("output_mode", G01_STRATEGIES)
    def test_unmasked_serialization_cycle(self, output_mode):
        """Full `.keras` round trip for the three strategies, plus a config round trip."""
        config = {**G01_CFG, 'output_mode': output_mode}
        input_ids = _g01_np(ops.convert_to_tensor(_g01_ids(seed=23)))

        inputs = layers.Input(shape=(G01_S,), dtype='int32')
        outputs = TextEncoder(**config)(inputs)
        model = models.Model(inputs, outputs)
        original = _g01_np(model(input_ids, training=False))

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, f"text_encoder_{output_mode}.keras")
            model.save(filepath)
            loaded = models.load_model(filepath)
            np.testing.assert_allclose(
                _g01_np(loaded(input_ids, training=False)), original,
                rtol=1e-6, atol=1e-6,
                err_msg=f"output_mode={output_mode!r} failed its .keras round trip",
            )

        encoder = TextEncoder(**config)
        rebuilt = TextEncoder.from_config(encoder.get_config())
        assert rebuilt.output_mode == output_mode
        assert rebuilt.pooling_layer.strategy == encoder.pooling_layer.strategy
