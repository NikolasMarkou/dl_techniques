"""
Comprehensive pytest test suite for the BERT Embeddings layer.

This module provides extensive testing for the Embeddings layer, ensuring it adheres
to modern Keras 3 best practices as outlined in the provided guide. Tests include:
- Initialization and parameter validation.
- Correct build process, including explicit sub-layer building.
- Forward pass functionality with different input combinations.
- Behavior in training vs. inference modes.
- Support for various normalization types.
- A full serialization and deserialization cycle to guarantee production readiness.
- Completeness of the get_config method.
"""

import pytest
import numpy as np
import keras
from keras import ops
import tempfile
import os
from typing import Dict, Any

from dl_techniques.layers.embedding.bert_embeddings import BertEmbeddings


class TestEmbeddingsLayer:
    """Comprehensive test suite for the Embeddings layer."""

    @pytest.fixture
    def basic_params(self) -> Dict[str, Any]:
        """Provides a standard set of parameters for creating the layer."""
        return {
            'vocab_size': 1000,
            'hidden_size': 256,
            'max_position_embeddings': 128,
            'type_vocab_size': 2,
            'initializer_range': 0.02,
            'layer_norm_eps': 1e-12,
            'dropout_rate': 0.1,
            'normalization_type': 'layer_norm'
        }

    @pytest.fixture
    def sample_input(self, basic_params) -> keras.KerasTensor:
        """Provides a sample input tensor for testing forward passes."""
        batch_size, seq_length = 4, 32
        return ops.cast(
            keras.random.uniform(
                (batch_size, seq_length),
                minval=1,
                maxval=basic_params['vocab_size']
            ),
            dtype='int32'
        )

    def test_initialization(self, basic_params):
        """Test that the layer initializes correctly and creates sub-layers."""
        layer = BertEmbeddings(**basic_params)

        # Verify all parameters are stored correctly
        for key, value in basic_params.items():
            assert getattr(layer, key) == value

        # Verify sub-layers are created (but not built)
        assert isinstance(layer.word_embeddings, keras.layers.Embedding)
        assert isinstance(layer.position_embeddings, keras.layers.Embedding)
        assert isinstance(layer.token_type_embeddings, keras.layers.Embedding)
        assert isinstance(layer.layer_norm, keras.layers.LayerNormalization)
        assert isinstance(layer.dropout, keras.layers.Dropout)

        # The layer and its children should be unbuilt after initialization
        assert not layer.built
        assert not layer.word_embeddings.built

    def test_parameter_validation(self, basic_params):
        """Test that invalid __init__ parameters raise ValueErrors."""
        # Test invalid numerical parameters
        with pytest.raises(ValueError, match="vocab_size must be positive"):
            BertEmbeddings(**{**basic_params, 'vocab_size': 0})
        with pytest.raises(ValueError, match="hidden_size must be positive"):
            BertEmbeddings(**{**basic_params, 'hidden_size': -1})
        with pytest.raises(ValueError, match="initializer_range must be positive"):
            BertEmbeddings(**{**basic_params, 'initializer_range': 0})

        # Test invalid dropout probability
        with pytest.raises(ValueError, match="dropout_rate must be between 0 and 1"):
            BertEmbeddings(**{**basic_params, 'dropout_rate': 1.1})
        with pytest.raises(ValueError, match="dropout_rate must be between 0 and 1"):
            BertEmbeddings(**{**basic_params, 'dropout_rate': -0.1})

        # Test invalid normalization type
        with pytest.raises(ValueError, match="normalization_type must be one of"):
            BertEmbeddings(**{**basic_params, 'normalization_type': 'invalid_norm'})

    def test_build_process(self, basic_params, sample_input):
        """Verify the explicit build method correctly builds all sub-layers."""
        layer = BertEmbeddings(**basic_params)
        input_shape = ops.shape(sample_input)

        # Manually build the layer
        layer.build(input_shape)

        # The layer and all its sub-layers must be marked as built
        assert layer.built
        assert layer.word_embeddings.built
        assert layer.position_embeddings.built
        assert layer.token_type_embeddings.built
        assert layer.layer_norm.built
        assert layer.dropout.built

    def test_build_invalid_shape(self, basic_params):
        """Test that building with an invalid input shape raises an error."""
        layer = BertEmbeddings(**basic_params)
        with pytest.raises(ValueError, match="Expected 2D input shape"):
            layer.build((None, 32, 128))  # Invalid 3D shape

    def test_forward_pass_input_ids_only(self, basic_params, sample_input):
        """Test forward pass with only input_ids, relying on default creation."""
        layer = BertEmbeddings(**basic_params)
        output = layer(sample_input, training=False)

        expected_shape = (*ops.shape(sample_input), basic_params['hidden_size'])
        assert output.shape == expected_shape

    def test_forward_pass_with_token_type_ids(self, basic_params, sample_input):
        """Test forward pass with provided token_type_ids."""
        layer = BertEmbeddings(**basic_params)
        token_type_ids = ops.zeros_like(sample_input, dtype='int32')
        output = layer(sample_input, token_type_ids=token_type_ids, training=False)

        expected_shape = (*ops.shape(sample_input), basic_params['hidden_size'])
        assert output.shape == expected_shape

    def test_forward_pass_with_position_ids(self, basic_params, sample_input):
        """Test forward pass with provided position_ids."""
        layer = BertEmbeddings(**basic_params)
        position_ids = ops.arange(ops.shape(sample_input)[1], dtype='int32')
        position_ids = ops.broadcast_to(ops.expand_dims(position_ids, 0), ops.shape(sample_input))
        output = layer(sample_input, position_ids=position_ids, training=False)

        expected_shape = (*ops.shape(sample_input), basic_params['hidden_size'])
        assert output.shape == expected_shape

    def test_training_mode(self, basic_params, sample_input):
        """Test that dropout is applied in training mode but not evaluation."""
        # With dropout enabled
        layer_with_dropout = BertEmbeddings(**basic_params)
        output_train = layer_with_dropout(sample_input, training=True)
        output_eval = layer_with_dropout(sample_input, training=False)
        # Outputs should differ due to dropout
        assert not np.allclose(
            ops.convert_to_numpy(output_train),
            ops.convert_to_numpy(output_eval)
        )

        # With dropout disabled
        params_no_dropout = {**basic_params, 'dropout_rate': 0.0}
        layer_no_dropout = BertEmbeddings(**params_no_dropout)
        output_train_no_dropout = layer_no_dropout(sample_input, training=True)
        output_eval_no_dropout = layer_no_dropout(sample_input, training=False)
        # Outputs should be identical
        np.testing.assert_allclose(
            ops.convert_to_numpy(output_train_no_dropout),
            ops.convert_to_numpy(output_eval_no_dropout)
        )

    @pytest.mark.parametrize("norm_type", ['layer_norm', 'rms_norm', 'band_rms', 'batch_norm'])
    def test_normalization_types(self, basic_params, sample_input, norm_type):
        """Test that all supported normalization types are functional."""
        params = {**basic_params, 'normalization_type': norm_type}
        layer = BertEmbeddings(**params)
        output = layer(sample_input, training=False)

        expected_shape = (*ops.shape(sample_input), basic_params['hidden_size'])
        assert output.shape == expected_shape
        assert ops.all(ops.isfinite(output))

    def test_compute_output_shape(self, basic_params, sample_input):
        """Verify the compute_output_shape method."""
        layer = BertEmbeddings(**basic_params)
        input_shape = ops.shape(sample_input)
        output_shape = layer.compute_output_shape(input_shape)

        expected_shape = (*input_shape, basic_params['hidden_size'])
        assert output_shape == expected_shape

    def test_get_config_completeness(self, basic_params):
        """Verify that get_config includes all __init__ parameters."""
        layer = BertEmbeddings(**basic_params)
        config = layer.get_config()

        for key in basic_params:
            assert key in config, f"Key '{key}' is missing from get_config()"
            assert config[key] == basic_params[key]

    def test_serialization_cycle(self, basic_params, sample_input):
        """CRITICAL TEST: Ensure a full save and load cycle works perfectly."""
        # 1. Create original layer in a model
        inputs = keras.Input(shape=sample_input.shape[1:], dtype='int32')
        layer_output = BertEmbeddings(**basic_params)(inputs)
        model = keras.Model(inputs, layer_output)

        # 2. Get prediction from original model
        original_prediction = model(sample_input)

        # 3. Save and load the model
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_embeddings_model.keras')
            model.save(filepath)

            # The `custom_objects` argument is not needed due to the registration decorator
            loaded_model = keras.models.load_model(filepath)
            loaded_prediction = loaded_model(sample_input)

            # 4. Verify that predictions are identical
            np.testing.assert_allclose(
                ops.convert_to_numpy(original_prediction),
                ops.convert_to_numpy(loaded_prediction),
                rtol=1e-6, atol=1e-6,
                err_msg="Predictions differ after a save/load cycle."
            )

class TestOptionalBranches:
    """Tests for the three opt-in parameters added for the DistilBERT de-duplication.

    ``use_token_type_embeddings``, ``position_embedding_type`` and ``mask_zero``
    all default to the pre-existing BERT behaviour. Every test here asserts the
    parameter's EFFECT -- a weight set, an output value, an auto-mask, a raise --
    never merely that construction succeeded.
    """

    @pytest.fixture
    def distil_params(self) -> Dict[str, Any]:
        """Parameters mirroring a DistilBERT-style embedding stage."""
        return {
            'vocab_size': 512,
            'hidden_size': 64,
            'max_position_embeddings': 32,
            'type_vocab_size': None,
            'initializer_range': 0.02,
            'layer_norm_eps': 1e-12,
            'dropout_rate': 0.0,
            'normalization_type': 'layer_norm',
            'use_token_type_embeddings': False,
            'position_embedding_type': 'sinusoidal',
            'mask_zero': False,
        }

    @pytest.fixture
    def ids(self) -> keras.KerasTensor:
        """A fixed (2, 8) integer input carrying pad ids (``0``) in row 0."""
        return ops.convert_to_tensor(
            np.array([[5, 9, 11, 4, 0, 0, 0, 0],
                      [1, 2, 3, 4, 5, 6, 7, 8]], dtype='int32')
        )

    # -- use_token_type_embeddings -------------------------------------------------

    def test_token_type_disabled_creates_no_weight(self, distil_params, ids):
        """Disabling token types must remove the sub-layer AND its weight."""
        layer = BertEmbeddings(**distil_params)
        assert layer.token_type_embeddings is None

        output = layer(ids, training=False)
        assert output.shape == (2, 8, distil_params['hidden_size'])
        assert bool(ops.all(ops.isfinite(output)))

        paths = [w.path for w in layer.weights]
        assert not any('token_type_embeddings' in p for p in paths), paths
        # The enabled control must, by contrast, own exactly that weight.
        enabled = BertEmbeddings(**{**distil_params,
                                    'use_token_type_embeddings': True,
                                    'type_vocab_size': 2})
        enabled(ids, training=False)
        assert any('token_type_embeddings' in w.path for w in enabled.weights)

    def test_token_type_disabled_rejects_token_type_ids(self, distil_params, ids):
        """Passing token_type_ids to a layer with no segment embedding must raise."""
        layer = BertEmbeddings(**distil_params)
        with pytest.raises(ValueError, match="use_token_type_embeddings is False"):
            layer(ids, token_type_ids=ops.zeros_like(ids), training=False)

    def test_type_vocab_size_required_when_token_types_enabled(self, distil_params):
        """type_vocab_size is conditionally required, not unconditionally optional."""
        params = {**distil_params, 'use_token_type_embeddings': True}
        with pytest.raises(ValueError, match="type_vocab_size must be positive"):
            BertEmbeddings(**{**params, 'type_vocab_size': None})
        with pytest.raises(ValueError, match="type_vocab_size must be positive"):
            BertEmbeddings(**{**params, 'type_vocab_size': 0})

    def test_type_vocab_size_normalized_away_when_disabled(self, distil_params):
        """An inert type_vocab_size must never reach get_config() or the weights."""
        layer = BertEmbeddings(**{**distil_params, 'type_vocab_size': 7})
        assert layer.type_vocab_size is None
        assert layer.get_config()['type_vocab_size'] is None
        assert layer.token_type_embeddings is None

    # -- position_embedding_type ---------------------------------------------------

    def test_invalid_position_embedding_type_raises(self, distil_params):
        """An unknown value must RAISE -- never silently fall back to 'learned'."""
        with pytest.raises(ValueError, match="position_embedding_type must be one of"):
            BertEmbeddings(**{**distil_params, 'position_embedding_type': 'rotary'})

    def test_sinusoidal_odd_hidden_size_raises(self, distil_params):
        """The interleave needs sin/cos pairs, so an odd hidden_size must raise."""
        with pytest.raises(ValueError, match="hidden_size must be even"):
            BertEmbeddings(**{**distil_params, 'hidden_size': 65})
        # Control: the same odd hidden_size is legal on the learned branch.
        BertEmbeddings(**{**distil_params, 'hidden_size': 65,
                          'position_embedding_type': 'learned'})

    def test_sinusoidal_creates_no_position_weight(self, distil_params, ids):
        """The sinusoidal table is fixed: it must allocate no trainable weight."""
        layer = BertEmbeddings(**distil_params)
        layer(ids, training=False)
        assert layer.position_embeddings is None
        assert not any('position_embeddings' in w.path for w in layer.weights)

        learned = BertEmbeddings(**{**distil_params, 'position_embedding_type': 'learned'})
        learned(ids, training=False)
        assert any('position_embeddings' in w.path for w in learned.weights)

    def test_sinusoidal_table_matches_the_closed_form(self, distil_params, ids):
        """VALUE check of the sin/cos table against an independently written oracle."""
        layer = BertEmbeddings(**distil_params)
        table = ops.convert_to_numpy(
            layer._sinusoidal_position_embeddings(
                ops.broadcast_to(ops.expand_dims(ops.arange(8, dtype='int32'), 0), (2, 8)),
                'float32'
            )
        )

        # Oracle written from PE(p, 2i) = sin(p / 10000^(2i/d)),
        # PE(p, 2i+1) = cos(p / 10000^(2i/d)) -- not from the implementation.
        d = distil_params['hidden_size']
        angles = np.arange(8)[:, None] * np.exp(
            np.arange(0, d, 2) * -(np.log(10000.0) / d)
        )
        oracle = np.zeros((8, d), dtype='float64')
        oracle[:, 0::2] = np.sin(angles)
        oracle[:, 1::2] = np.cos(angles)

        np.testing.assert_allclose(table[0], oracle, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(table[1], oracle, rtol=1e-5, atol=1e-6)
        # Position 0 pins the interleave orientation: sin(0)=0, cos(0)=1.
        np.testing.assert_allclose(table[0, 0, 0::2], np.zeros(d // 2), atol=1e-7)
        np.testing.assert_allclose(table[0, 0, 1::2], np.ones(d // 2), atol=1e-7)

    def test_sinusoidal_differs_from_learned(self, distil_params, ids):
        """The branch must actually change the output, not just the weight set."""
        keras.utils.set_random_seed(1234)
        sinus = BertEmbeddings(**distil_params)(ids, training=False)
        keras.utils.set_random_seed(1234)
        learned = BertEmbeddings(
            **{**distil_params, 'position_embedding_type': 'learned'}
        )(ids, training=False)
        assert not np.allclose(ops.convert_to_numpy(sinus),
                               ops.convert_to_numpy(learned), atol=1e-5)

    def test_sinusoidal_forward_float32(self, distil_params, ids):
        """Sinusoidal forward under the default float32 policy."""
        assert keras.mixed_precision.global_policy().name == 'float32'
        output = BertEmbeddings(**distil_params)(ids, training=False)
        assert keras.backend.standardize_dtype(output.dtype) == 'float32'
        assert bool(ops.all(ops.isfinite(output)))

    def test_sinusoidal_forward_mixed_float16(self, distil_params, ids):
        """Sinusoidal forward under mixed_float16.

        The pre-fix defect was a hard float32 table summed with a float16 word
        embedding, raising ``InvalidArgumentError: cannot compute AddV2 ... expected
        to be a half tensor but is a float tensor``. Executing the forward pass is
        the only way to observe the fix -- reading the cast is not.
        """
        previous = keras.mixed_precision.global_policy()
        try:
            keras.mixed_precision.set_global_policy('mixed_float16')
            layer = BertEmbeddings(**distil_params)
            output = layer(ids, training=False)
            assert keras.backend.standardize_dtype(output.dtype) == 'float16'
            assert bool(ops.all(ops.isfinite(output)))
            assert not np.allclose(ops.convert_to_numpy(output), 0.0)
        finally:
            keras.mixed_precision.set_global_policy(previous)

    def test_sinusoidal_accepts_positions_beyond_max_position_embeddings(
            self, distil_params):
        """The sinusoidal branch is unbounded by construction; assert it, don't assume."""
        layer = BertEmbeddings(**distil_params)
        long_ids = ops.convert_to_tensor(
            np.ones((1, distil_params['max_position_embeddings'] + 16), dtype='int32')
        )
        output = layer(long_ids, training=False)
        assert output.shape == (1, distil_params['max_position_embeddings'] + 16,
                                distil_params['hidden_size'])
        assert bool(ops.all(ops.isfinite(output)))

    # -- mask_zero -----------------------------------------------------------------

    def test_mask_zero_controls_auto_masking(self, distil_params, ids):
        """mask_zero must decide whether a Keras auto-mask is produced at all."""
        off = BertEmbeddings(**distil_params)
        off(ids, training=False)
        assert off.word_embeddings.mask_zero is False
        assert off.word_embeddings.compute_mask(ids) is None

        on = BertEmbeddings(**{**distil_params, 'mask_zero': True})
        on(ids, training=False)
        assert on.word_embeddings.mask_zero is True
        mask = on.word_embeddings.compute_mask(ids)
        assert mask is not None
        np.testing.assert_array_equal(
            ops.convert_to_numpy(mask),
            ops.convert_to_numpy(ids) != 0
        )

    # -- serialization -------------------------------------------------------------

    def test_get_config_carries_the_new_keys(self, distil_params):
        """All three new keys must appear in get_config() with their real VALUES."""
        layer = BertEmbeddings(**distil_params)
        config = layer.get_config()
        for key in ('use_token_type_embeddings', 'position_embedding_type', 'mask_zero'):
            assert key in config, f"Key '{key}' is missing from get_config()"
            assert config[key] == distil_params[key]

    def test_new_config_defaults_preserve_bert_behaviour(self):
        """At the BERT call site the three keys must sit at their legacy values."""
        config = BertEmbeddings(
            vocab_size=128, hidden_size=32, max_position_embeddings=16,
            type_vocab_size=2
        ).get_config()
        assert config['use_token_type_embeddings'] is True
        assert config['position_embedding_type'] == 'learned'
        assert config['mask_zero'] is True

    def test_round_trip_of_non_default_config(self, distil_params, ids):
        """A .keras round trip must preserve the new keys AND the forward VALUES."""
        inputs = keras.Input(shape=(8,), dtype='int32')
        model = keras.Model(inputs, BertEmbeddings(**distil_params)(inputs))
        original = model(ids)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'distil_style_embeddings.keras')
            model.save(filepath)
            loaded_model = keras.models.load_model(filepath)

        loaded_layer = [l for l in loaded_model.layers
                        if isinstance(l, BertEmbeddings)][0]
        loaded_config = loaded_layer.get_config()
        for key in ('use_token_type_embeddings', 'position_embedding_type',
                    'mask_zero', 'type_vocab_size'):
            assert loaded_config[key] == distil_params[key]
        assert loaded_layer.token_type_embeddings is None
        assert loaded_layer.position_embeddings is None

        np.testing.assert_allclose(
            ops.convert_to_numpy(original),
            ops.convert_to_numpy(loaded_model(ids)),
            rtol=1e-6, atol=1e-6,
            err_msg="Predictions differ after a save/load cycle."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
