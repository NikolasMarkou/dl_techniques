import pytest
import numpy as np
import tensorflow as tf
import keras
from keras import ops, layers, models
import tempfile
import os
from typing import Any, Dict

from dl_techniques.layers.text_decoder import TextDecoder


class TestTextDecoder:
    """
    Comprehensive test suite for the configurable TextDecoder.
    """

    @pytest.fixture
    def basic_config(self) -> Dict[str, Any]:
        """Provides a basic configuration for a small, testable decoder."""
        return {
            'vocab_size': 1000,
            'embed_dim': 64,
            'depth': 2,
            'num_heads': 4,
            'max_seq_len': 16,
        }

    @pytest.fixture
    def modern_config(self) -> Dict[str, Any]:
        """Provides a modern decoder configuration."""
        return {
            'vocab_size': 32000,
            'embed_dim': 512,
            'depth': 4,
            'num_heads': 8,
            'max_seq_len': 128,
            'positional_type': 'sincos',
            'normalization_type': 'rms_norm',
            'normalization_position': 'pre',
            'ffn_type': 'swiglu',
            'stochastic_depth_rate': 0.1,
        }

    @pytest.fixture
    def sample_input_ids(self) -> tf.Tensor:
        """Provides sample input token IDs for testing."""
        return tf.random.uniform(shape=(2, 16), minval=0, maxval=1000, dtype=tf.int32)

    @pytest.fixture
    def sample_attention_mask(self) -> tf.Tensor:
        """Provides a sample attention mask with padding."""
        mask = tf.ones((2, 16), dtype=tf.int32)
        # Pad the last 4 tokens of the second sequence
        mask = tf.tensor_scatter_nd_update(mask, [[1, 12], [1, 13], [1, 14], [1, 15]], [0, 0, 0, 0])
        return mask

    # ===============================================
    # 1. Initialization and Build Tests
    # ===============================================
    def test_initialization_defaults(self, basic_config):
        """Tests decoder initialization with default parameters."""
        decoder = TextDecoder(**basic_config)
        assert not decoder.built
        assert decoder.embedding_type == 'learned'
        assert decoder.positional_type == 'learned'
        assert decoder.normalization_position == 'post'
        assert decoder.ffn_type == 'mlp'

    def test_build_process(self, basic_config, sample_input_ids):
        """Tests that the decoder and its sub-layers are built correctly."""
        decoder = TextDecoder(**basic_config)
        assert not decoder.built
        output = decoder(sample_input_ids)
        assert decoder.built
        assert decoder.word_embeddings.built
        assert decoder.positional_embeddings.built
        assert all(layer.built for layer in decoder.decoder_layers)

    # ===============================================
    # 2. Parameter Validation Tests
    # ===============================================
    def test_invalid_embed_dim_num_heads(self, basic_config):
        """Tests validation for embed_dim and num_heads compatibility."""
        config = {**basic_config, 'num_heads': 5}
        with pytest.raises(ValueError, match="must be divisible by"):
            TextDecoder(**config)

    @pytest.mark.parametrize("param, value", [("vocab_size", 0), ("embed_dim", -1), ("depth", 0)])
    def test_invalid_dimensions(self, basic_config, param, value):
        """Tests that positive dimensions are enforced."""
        config = {**basic_config, param: value}
        with pytest.raises(ValueError, match="must be positive"):
            TextDecoder(**config)

    # ===============================================
    # 3. Forward Pass and Causal Masking Tests
    # ===============================================
    def test_forward_pass_shape(self, basic_config, sample_input_ids):
        """Tests the output shape of a forward pass."""
        decoder = TextDecoder(**basic_config)
        output = decoder(sample_input_ids)
        expected_shape = (
            sample_input_ids.shape[0],
            sample_input_ids.shape[1],
            basic_config['embed_dim']
        )
        assert output.shape == expected_shape

    def test_causal_masking_behavior(self, basic_config):
        """Verifies that the output at a timestep is not affected by future timesteps."""
        decoder = TextDecoder(**basic_config)

        input_ids = tf.constant([[10, 20, 30, 40]], dtype=tf.int32)
        input_ids_modified = tf.constant([[10, 20, 30, 99]], dtype=tf.int32)  # Change last token

        # Call build explicitly to initialize weights
        decoder.build(input_ids.shape)

        # Get weights to ensure both calls use the same model state
        weights = decoder.get_weights()

        # Create two identical instances to ensure state is not shared across calls
        decoder1 = TextDecoder(**basic_config)
        decoder1.build(input_ids.shape)
        decoder1.set_weights(weights)

        decoder2 = TextDecoder(**basic_config)
        decoder2.build(input_ids_modified.shape)
        decoder2.set_weights(weights)

        output_original = decoder1(input_ids, training=False)
        output_modified = decoder2(input_ids_modified, training=False)

        # The output at timestep 2 (index 2) should be identical
        np.testing.assert_allclose(
            ops.convert_to_numpy(output_original[:, 2, :]),
            ops.convert_to_numpy(output_modified[:, 2, :]),
            rtol=1e-6, atol=1e-6
        )

        # The output at timestep 3 (index 3) should be different
        assert not np.allclose(
            ops.convert_to_numpy(output_original[:, 3, :]),
            ops.convert_to_numpy(output_modified[:, 3, :])
        )

    def test_padding_mask_integration(self, basic_config, sample_input_ids, sample_attention_mask):
        """Tests that padding masks are correctly integrated with the causal mask."""
        decoder = TextDecoder(**basic_config)

        # A forward pass should work without error
        output = decoder(sample_input_ids, attention_mask=sample_attention_mask)
        expected_shape = (
            sample_input_ids.shape[0],
            sample_input_ids.shape[1],
            basic_config['embed_dim']
        )
        assert output.shape == expected_shape

    # ===============================================
    # 4. Serialization Tests
    # ===============================================
    def test_full_serialization_cycle_basic(self, basic_config, sample_input_ids):
        """Tests the full save/load cycle with a basic configuration."""
        inputs = layers.Input(shape=sample_input_ids.shape[1:], dtype='int32')
        outputs = TextDecoder(**basic_config)(inputs)
        model = models.Model(inputs, outputs)

        original_prediction = model(sample_input_ids, training=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_basic_decoder.keras")
            model.save(filepath)
            loaded_model = models.load_model(filepath)
            loaded_prediction = loaded_model(sample_input_ids, training=False)

            np.testing.assert_allclose(
                ops.convert_to_numpy(original_prediction),
                ops.convert_to_numpy(loaded_prediction),
                rtol=1e-6, atol=1e-6
            )

    def test_full_serialization_cycle_modern(self, modern_config):
        """Tests the full save/load cycle with a modern configuration."""
        seq_len = modern_config['max_seq_len']
        sample_input_ids = tf.random.uniform((2, seq_len), maxval=1000, dtype=tf.int32)

        inputs = layers.Input(shape=(seq_len,), dtype='int32')
        outputs = TextDecoder(**modern_config)(inputs)
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
                rtol=1e-5, atol=1e-5  # Looser tolerance for complex models
            )

    # ===============================================
    # 5. Training Integration Tests
    # ===============================================
    def test_gradient_flow(self, basic_config, sample_input_ids):
        """Tests that gradients flow through the decoder."""
        decoder = TextDecoder(**basic_config)

        with tf.GradientTape() as tape:
            output = decoder(sample_input_ids, training=True)
            loss = ops.mean(ops.square(output))

        gradients = tape.gradient(loss, decoder.trainable_variables)
        assert len(gradients) > 0, "No gradients were computed."
        assert all(g is not None for g in gradients), "A gradient is None."

    def test_model_training_loop_integration(self, basic_config):
        """Tests integration into a standard training loop."""
        seq_len = basic_config['max_seq_len']
        vocab_size = basic_config['vocab_size']

        model = models.Sequential([
            layers.InputLayer(shape=(seq_len,), dtype='int32'),
            TextDecoder(**basic_config),
            layers.Dense(vocab_size)
        ])

        model.compile("adam", keras.losses.SparseCategoricalCrossentropy(from_logits=True))

        x_train = tf.random.uniform((8, seq_len), maxval=vocab_size, dtype=tf.int32)
        y_train = tf.random.uniform((8, seq_len), maxval=vocab_size, dtype=tf.int32)

        history = model.fit(x_train, y_train, epochs=1, batch_size=4, verbose=0)
        assert 'loss' in history.history
        assert not np.isnan(history.history['loss'][0])