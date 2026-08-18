import pytest
import tempfile
import os
import numpy as np
import keras
import tensorflow as tf
from typing import Any, Dict


from dl_techniques.models.mobile_clip.components import (
    ImageProjectionHead,
    MobileClipImageEncoder,
    MobileClipTextEncoder
)

from ..knob_sensitivity_oracle import assert_value_knob_changes_output

class TestImageProjectionHead:
    """Comprehensive test suite for ImageProjectionHead layer."""

    @pytest.fixture
    def layer_config(self) -> Dict[str, Any]:
        """Standard configuration for testing."""
        return {
            'projection_dim': 512,
            'dropout_rate': 0.1,
            'activation': 'relu'
        }

    @pytest.fixture
    def sample_input(self) -> keras.KerasTensor:
        """Sample input for testing - 4D feature maps."""
        return keras.random.normal(shape=(4, 7, 7, 2048))

    def test_initialization(self, layer_config):
        """Test layer initialization."""
        layer = ImageProjectionHead(**layer_config)

        assert hasattr(layer, 'projection_dim')
        assert hasattr(layer, 'dropout_rate')
        assert hasattr(layer, 'activation')
        assert not layer.built
        assert layer.global_pool is not None  # Sub-layers created
        assert layer.dropout is not None
        assert layer.projection is not None

    def test_forward_pass(self, layer_config, sample_input):
        """Test forward pass and building."""
        layer = ImageProjectionHead(**layer_config)

        output = layer(sample_input)

        assert layer.built
        assert output.shape[0] == sample_input.shape[0]  # Batch size preserved
        assert output.shape[1] == layer_config['projection_dim']  # Correct output dim
        assert len(output.shape) == 2  # Flattened to 2D

    def test_serialization_cycle(self, layer_config, sample_input):
        """CRITICAL TEST: Full serialization cycle."""
        # Create model with custom layer
        inputs = keras.Input(shape=sample_input.shape[1:])
        outputs = ImageProjectionHead(**layer_config)(inputs)
        model = keras.Model(inputs, outputs)

        # Get original prediction
        original_pred = model(sample_input)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_model.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_pred = loaded_model(sample_input)

            # Verify identical predictions
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_pred),
                keras.ops.convert_to_numpy(loaded_pred),
                rtol=1e-6, atol=1e-6,
                err_msg="Predictions differ after serialization"
            )

    def test_config_completeness(self, layer_config):
        """Test that get_config contains all __init__ parameters."""
        layer = ImageProjectionHead(**layer_config)
        config = layer.get_config()

        # Check all config parameters are present
        for key in layer_config:
            assert key in config, f"Missing {key} in get_config()"

        # Check additional parameters that have defaults
        expected_keys = {
            'projection_dim', 'dropout_rate', 'activation',
            'kernel_initializer', 'bias_initializer'
        }
        for key in expected_keys:
            assert key in config, f"Missing {key} in get_config()"

    def test_gradients_flow(self, layer_config, sample_input):
        """Test gradient computation."""
        layer = ImageProjectionHead(**layer_config)

        with tf.GradientTape() as tape:
            tape.watch(sample_input)
            output = layer(sample_input)
            loss = keras.ops.mean(keras.ops.square(output))

        gradients = tape.gradient(loss, layer.trainable_variables)

        assert all(g is not None for g in gradients)
        assert len(gradients) > 0

    @pytest.mark.parametrize("training", [True, False, None])
    def test_training_modes(self, layer_config, sample_input, training):
        """Test behavior in different training modes."""
        layer = ImageProjectionHead(**layer_config)

        output = layer(sample_input, training=training)
        assert output.shape[0] == sample_input.shape[0]
        assert output.shape[1] == layer_config['projection_dim']

    def test_edge_cases(self):
        """Test error conditions."""
        with pytest.raises(ValueError):
            ImageProjectionHead(projection_dim=0)  # Invalid projection_dim

        with pytest.raises(ValueError):
            ImageProjectionHead(projection_dim=512, dropout_rate=-0.1)  # Invalid dropout

        with pytest.raises(ValueError):
            ImageProjectionHead(projection_dim=512, dropout_rate=1.5)  # Invalid dropout

    def test_different_activations(self, sample_input):
        """The activation must reach the projection head's forward pass.

        A value knob: the head holds the same weights for every activation, so
        under one seed the layers are bit-identical and the whole output
        difference is the activation. ``None`` (linear) is included, which is
        the setting a dropped kwarg would silently collapse the others onto.
        """
        activations_to_test = [None, 'relu', 'gelu', 'tanh']

        def _build(activation):
            layer = ImageProjectionHead(projection_dim=256, activation=activation)
            layer(keras.ops.zeros_like(sample_input[:1]))
            return layer

        builders = {a: (lambda a=a: _build(a)) for a in activations_to_test}
        deltas = assert_value_knob_changes_output(
            builders, sample_input, knob="activation",
        )
        assert min(deltas.values()) > 1e-5

        for activation in activations_to_test:
            layer = ImageProjectionHead(projection_dim=256, activation=activation)
            output = layer(sample_input)
            assert output.shape == (sample_input.shape[0], 256)

    def test_compute_output_shape(self, layer_config):
        """Test output shape computation."""
        layer = ImageProjectionHead(**layer_config)
        input_shape = (32, 7, 7, 2048)

        output_shape = layer.compute_output_shape(input_shape)
        expected_shape = (32, layer_config['projection_dim'])

        assert output_shape == expected_shape


class TestMobileClipImageEncoder:
    """Comprehensive test suite for MobileClipImageEncoder model."""

    @pytest.fixture
    def model_config(self) -> Dict[str, Any]:
        """Standard configuration for testing."""
        return {
            'backbone_name': 'MobileNetV2',
            'projection_dim': 512,
            'backbone_weights': None,  # Use None to avoid downloading weights
            'backbone_trainable': True,
            'projection_dropout': 0.1
        }

    @pytest.fixture
    def sample_input(self) -> keras.KerasTensor:
        """Sample input for testing - RGB images."""
        return keras.random.normal(shape=(4, 224, 224, 3))

    def test_initialization(self, model_config):
        """Test model initialization."""
        model = MobileClipImageEncoder(**model_config)

        assert hasattr(model, 'backbone_name')
        assert hasattr(model, 'projection_dim')
        assert hasattr(model, 'backbone')
        assert hasattr(model, 'projection_head')
        assert model.backbone is not None
        assert model.projection_head is not None

    def test_forward_pass(self, model_config, sample_input):
        """Test forward pass."""
        model = MobileClipImageEncoder(**model_config)

        output = model(sample_input)

        assert output.shape[0] == sample_input.shape[0]  # Batch size preserved
        assert output.shape[1] == model_config['projection_dim']  # Correct output dim
        assert len(output.shape) == 2  # 2D embeddings

    def test_serialization_cycle(self, model_config, sample_input):
        """CRITICAL TEST: Full serialization cycle."""
        # Create model
        model = MobileClipImageEncoder(**model_config)

        # Get original prediction
        original_pred = model(sample_input)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_model.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_pred = loaded_model(sample_input)

            # Verify identical predictions
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_pred),
                keras.ops.convert_to_numpy(loaded_pred),
                rtol=1e-6, atol=1e-6,
                err_msg="Predictions differ after serialization"
            )

    def test_config_completeness(self, model_config):
        """Test that get_config contains all __init__ parameters."""
        model = MobileClipImageEncoder(**model_config)
        config = model.get_config()

        # Check all config parameters are present
        for key in model_config:
            assert key in config, f"Missing {key} in get_config()"

    def test_gradients_flow(self, model_config, sample_input):
        """Test gradient computation."""
        model = MobileClipImageEncoder(**model_config)

        with tf.GradientTape() as tape:
            tape.watch(sample_input)
            output = model(sample_input)
            loss = keras.ops.mean(keras.ops.square(output))

        gradients = tape.gradient(loss, model.trainable_variables)

        assert all(g is not None for g in gradients)
        assert len(gradients) > 0

    @pytest.mark.parametrize("training", [True, False, None])
    def test_training_modes(self, model_config, sample_input, training):
        """Test behavior in different training modes."""
        model = MobileClipImageEncoder(**model_config)

        output = model(sample_input, training=training)
        assert output.shape[0] == sample_input.shape[0]
        assert output.shape[1] == model_config['projection_dim']

    def test_edge_cases(self):
        """Test error conditions."""
        with pytest.raises(ValueError):
            MobileClipImageEncoder(projection_dim=0)  # Invalid projection_dim

        with pytest.raises(ValueError):
            MobileClipImageEncoder(projection_dim=512, projection_dropout=-0.1)  # Invalid dropout

        with pytest.raises(ValueError):
            MobileClipImageEncoder(backbone_name='NonexistentBackbone')  # Invalid backbone

    def test_backbone_trainable_setting(self, sample_input):
        """Test backbone trainable setting."""
        # Test trainable=True
        model_trainable = MobileClipImageEncoder(
            backbone_trainable=True,
            backbone_weights=None
        )
        assert model_trainable.backbone.trainable

        # Test trainable=False
        model_frozen = MobileClipImageEncoder(
            backbone_trainable=False,
            backbone_weights=None
        )
        assert not model_frozen.backbone.trainable


class TestMobileClipTextEncoder:
    """Comprehensive test suite for MobileClipTextEncoder layer."""

    @pytest.fixture
    def layer_config(self) -> Dict[str, Any]:
        """Standard configuration for testing."""
        return {
            'vocab_size': 10000,
            'max_seq_len': 128,
            'embed_dim': 512,
            'num_layers': 6,
            'num_heads': 8,
            'intermediate_size': 2048,
            'projection_dim': 512,
            'dropout_rate': 0.1,
            'attention_dropout_rate': 0.1
        }

    @pytest.fixture
    def sample_input(self) -> keras.KerasTensor:
        """Sample input for testing - text token IDs."""
        # Create tokens with EOT token at the end (highest ID)
        batch_size, seq_len = 4, 64
        tokens = keras.random.randint((batch_size, seq_len), 0, 9999, dtype="int32")
        # Set last token to vocab_size - 1 to simulate EOT
        tokens = keras.ops.concatenate([
            tokens[:, :-1],
            keras.ops.full((batch_size, 1), 9999, dtype="int32")
        ], axis=1)
        return tokens

    def test_initialization(self, layer_config):
        """Test layer initialization."""
        layer = MobileClipTextEncoder(**layer_config)

        assert hasattr(layer, 'vocab_size')
        assert hasattr(layer, 'max_seq_len')
        assert hasattr(layer, 'embed_dim')
        assert hasattr(layer, 'num_layers')
        assert not layer.built
        assert layer.token_embedding is not None  # Sub-layers created
        assert layer.positional_embedding is not None
        assert len(layer.transformer_layers) == layer_config['num_layers']

    def test_forward_pass(self, layer_config, sample_input):
        """Test forward pass and building."""
        layer = MobileClipTextEncoder(**layer_config)

        output = layer(sample_input)

        assert layer.built
        assert output.shape[0] == sample_input.shape[0]  # Batch size preserved
        assert output.shape[1] == layer_config['projection_dim']  # Correct output dim
        assert len(output.shape) == 2  # 2D embeddings

    def test_serialization_cycle(self, layer_config, sample_input):
        """CRITICAL TEST: Full serialization cycle."""
        # Create model with custom layer
        inputs = keras.Input(shape=sample_input.shape[1:])
        outputs = MobileClipTextEncoder(**layer_config)(inputs)
        model = keras.Model(inputs, outputs)

        # Get original prediction
        original_pred = model(sample_input)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_model.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_pred = loaded_model(sample_input)

            # Verify identical predictions
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_pred),
                keras.ops.convert_to_numpy(loaded_pred),
                rtol=1e-6, atol=1e-6,
                err_msg="Predictions differ after serialization"
            )

    def test_config_completeness(self, layer_config):
        """Test that get_config contains all __init__ parameters."""
        layer = MobileClipTextEncoder(**layer_config)
        config = layer.get_config()

        # Check all config parameters are present
        for key in layer_config:
            assert key in config, f"Missing {key} in get_config()"

        # Check additional parameters that have defaults
        expected_keys = {
            'vocab_size', 'max_seq_len', 'embed_dim', 'num_layers', 'num_heads',
            'intermediate_size', 'projection_dim', 'dropout_rate',
            'attention_dropout_rate', 'use_causal_mask', 'embed_scale'
        }
        for key in expected_keys:
            assert key in config, f"Missing {key} in get_config()"

    def test_gradients_flow(self, layer_config, sample_input):
        """Test gradient computation."""
        layer = MobileClipTextEncoder(**layer_config)

        with tf.GradientTape() as tape:
            tape.watch(sample_input)
            output = layer(sample_input)
            loss = keras.ops.mean(keras.ops.square(output))

        gradients = tape.gradient(loss, layer.trainable_variables)

        assert all(g is not None for g in gradients)
        assert len(gradients) > 0

    @pytest.mark.parametrize("training", [True, False, None])
    def test_training_modes(self, layer_config, sample_input, training):
        """Test behavior in different training modes."""
        layer = MobileClipTextEncoder(**layer_config)

        output = layer(sample_input, training=training)
        assert output.shape[0] == sample_input.shape[0]
        assert output.shape[1] == layer_config['projection_dim']

    def test_edge_cases(self):
        """Test error conditions."""
        base_config = {
            'vocab_size': 10000,
            'max_seq_len': 128,
            'embed_dim': 512,
            'num_layers': 6,
            'num_heads': 8,
            'intermediate_size': 2048,
            'projection_dim': 512
        }

        with pytest.raises(ValueError):
            MobileClipTextEncoder(**{**base_config, 'vocab_size': 0})  # Invalid vocab_size

        with pytest.raises(ValueError):
            MobileClipTextEncoder(**{**base_config, 'embed_dim': 513, 'num_heads': 8})  # Non-divisible

        with pytest.raises(ValueError):
            MobileClipTextEncoder(**{**base_config, 'dropout_rate': -0.1})  # Invalid dropout

    def test_causal_mask_setting(self, layer_config, sample_input):
        """Test causal mask functionality."""
        # Test with causal mask
        layer_causal = MobileClipTextEncoder(**{**layer_config, 'use_causal_mask': True})
        output_causal = layer_causal(sample_input)

        # Test without causal mask
        layer_no_causal = MobileClipTextEncoder(**{**layer_config, 'use_causal_mask': False})
        output_no_causal = layer_no_causal(sample_input)

        # Both should produce valid outputs with same shape
        assert output_causal.shape == output_no_causal.shape
        assert output_causal.shape[1] == layer_config['projection_dim']

    def test_embed_scale_setting(self, sample_input):
        """Test custom embed scale."""
        config = {
            'vocab_size': 10000,
            'max_seq_len': 128,
            'embed_dim': 512,
            'num_layers': 2,
            'num_heads': 8,
            'intermediate_size': 1024,
            'projection_dim': 256
        }

        # Test default embed scale
        layer_default = MobileClipTextEncoder(**config)
        assert layer_default.embed_scale == (config['embed_dim'] ** -0.5)

        # Test custom embed scale
        custom_scale = 0.1
        layer_custom = MobileClipTextEncoder(**{**config, 'embed_scale': custom_scale})
        assert layer_custom.embed_scale == custom_scale

    def test_compute_output_shape(self, layer_config):
        """Test output shape computation."""
        layer = MobileClipTextEncoder(**layer_config)
        input_shape = (32, 64)  # (batch_size, seq_len)

        output_shape = layer.compute_output_shape(input_shape)
        expected_shape = (32, layer_config['projection_dim'])

        assert output_shape == expected_shape


# Integration tests
class TestMobileClipIntegration:
    """Integration tests for Mobile CLIP components."""

    def test_image_text_embedding_compatibility(self):
        """Test that image and text encoders produce compatible embeddings."""
        projection_dim = 512

        # Create encoders
        image_encoder = MobileClipImageEncoder(
            projection_dim=projection_dim,
            backbone_weights=None
        )

        text_encoder = MobileClipTextEncoder(
            vocab_size=10000,
            max_seq_len=128,
            embed_dim=256,
            num_layers=4,
            num_heads=8,
            intermediate_size=1024,
            projection_dim=projection_dim
        )

        # Create sample inputs
        images = keras.random.normal((8, 224, 224, 3))
        batch_size, seq_len = 8, 64
        text_tokens = keras.random.randint((batch_size, seq_len), 0, 9999, dtype="int32")
        # Add EOT tokens
        text_tokens = keras.ops.concatenate([
            text_tokens[:, :-1],
            keras.ops.full((batch_size, 1), 9999, dtype="int32")
        ], axis=1)

        # Get embeddings
        image_embeddings = image_encoder(images)
        text_embeddings = text_encoder(text_tokens)

        # Check compatibility
        assert image_embeddings.shape == text_embeddings.shape
        assert image_embeddings.shape[1] == projection_dim

        # Test similarity computation (should not error)
        similarity = keras.ops.matmul(image_embeddings, keras.ops.transpose(text_embeddings))
        assert similarity.shape == (8, 8)

    def test_end_to_end_training_compatibility(self):
        """Test that components work together in training setup."""
        projection_dim = 256

        # Create models
        image_encoder = MobileClipImageEncoder(
            projection_dim=projection_dim,
            backbone_weights=None
        )

        text_encoder = MobileClipTextEncoder(
            vocab_size=5000,
            max_seq_len=64,
            embed_dim=256,
            num_layers=2,
            num_heads=8,
            intermediate_size=512,
            projection_dim=projection_dim
        )

        # Create sample batch
        batch_size = 4
        images = keras.random.normal((batch_size, 224, 224, 3))
        text_tokens = keras.random.randint((batch_size, 32), 0, 4999, dtype="int32")
        # Add EOT tokens
        text_tokens = keras.ops.concatenate([
            text_tokens[:, :-1],
            keras.ops.full((batch_size, 1), 4999, dtype="int32")
        ], axis=1)

        # Forward pass
        with tf.GradientTape() as tape:
            image_embeddings = image_encoder(images, training=True)
            text_embeddings = text_encoder(text_tokens, training=True)

            # Simple contrastive-style loss for testing
            similarity = keras.ops.matmul(image_embeddings, keras.ops.transpose(text_embeddings))
            # Dummy loss - just for gradient testing
            labels = keras.ops.eye(batch_size)
            loss = keras.ops.mean(keras.ops.square(similarity - labels))

        # Test gradients
        all_variables = image_encoder.trainable_variables + text_encoder.trainable_variables
        gradients = tape.gradient(loss, all_variables)

        assert all(g is not None for g in gradients)
        assert len(gradients) > 0

        # Test that embeddings have the expected properties
        assert keras.ops.all(keras.ops.isfinite(image_embeddings))
        assert keras.ops.all(keras.ops.isfinite(text_embeddings))


class TestTextEncoderCausalMaskGraphSafety:
    """The causal mask must survive tracing, not just eager execution.

    `MobileClipTextEncoder.call()` builds a causal mask when
    `use_causal_mask=True`. It used to do so with `keras.ops.tril`, which routes
    through a `tf.cond` that rejects a Python-bool predicate the moment it is
    traced::

        TypeError: pred must not be a Python bool

    That failure is EAGER-INVISIBLE: a direct call works, and every graph path
    fails (`tf.function`, `Model.predict` with a static or dynamic sequence
    axis, `.keras` save/load, `jit_compile=True`). Keras also downgrades a
    `call()` crash during build-tracing to a `UserWarning`, so this whole suite
    reported PASS while the exception text appeared in its own output. These
    tests turn the graph paths into real assertions.

    Every test below is parametrized over `use_causal_mask` so the
    mask-disabled path is a control: if a "fix" broke the encoder generally
    rather than the mask specifically, the `False` case fails too and the
    diagnosis is unambiguous.
    """

    ENC_KWARGS = {
        'vocab_size': 64,
        'max_seq_len': 16,
        'embed_dim': 32,
        'num_layers': 1,
        'num_heads': 4,
        'intermediate_size': 64,
        'projection_dim': 32,
    }
    BATCH, SEQ = 2, 16

    @classmethod
    def _encoder(cls, use_causal_mask: bool) -> MobileClipTextEncoder:
        return MobileClipTextEncoder(use_causal_mask=use_causal_mask, **cls.ENC_KWARGS)

    @classmethod
    def _tokens(cls) -> np.ndarray:
        return np.random.default_rng(0).integers(
            0, cls.ENC_KWARGS['vocab_size'], size=(cls.BATCH, cls.SEQ)
        ).astype('int32')

    @pytest.mark.parametrize('use_causal_mask', [True, False])
    def test_eager_call_still_works(self, use_causal_mask):
        """Control: the path that was ALWAYS green must stay green."""
        out = self._encoder(use_causal_mask)(keras.ops.convert_to_tensor(self._tokens()))
        assert out.shape == (self.BATCH, self.ENC_KWARGS['projection_dim'])
        assert keras.ops.all(keras.ops.isfinite(out))

    @pytest.mark.parametrize('use_causal_mask', [True, False])
    def test_traced_tf_function(self, use_causal_mask):
        """`tf.function` tracing — the minimal reproducer of the trap."""
        encoder = self._encoder(use_causal_mask)

        @tf.function
        def run(tokens):
            return encoder(tokens)

        out = run(tf.constant(self._tokens()))
        assert tuple(out.shape) == (self.BATCH, self.ENC_KWARGS['projection_dim'])

    @pytest.mark.parametrize('use_causal_mask', [True, False])
    def test_functional_model_predict_static_sequence(self, use_causal_mask):
        inputs = keras.Input(shape=(self.SEQ,), dtype='int32')
        model = keras.Model(inputs, self._encoder(use_causal_mask)(inputs))
        out = model.predict(self._tokens(), verbose=0)
        assert out.shape == (self.BATCH, self.ENC_KWARGS['projection_dim'])
        assert np.isfinite(out).all()

    @pytest.mark.parametrize('use_causal_mask', [True, False])
    def test_functional_model_predict_dynamic_sequence(self, use_causal_mask):
        """A `None` sequence axis makes the mask size symbolic at trace time."""
        inputs = keras.Input(shape=(None,), dtype='int32')
        model = keras.Model(inputs, self._encoder(use_causal_mask)(inputs))
        out = model.predict(self._tokens(), verbose=0)
        assert out.shape == (self.BATCH, self.ENC_KWARGS['projection_dim'])
        assert np.isfinite(out).all()

    @pytest.mark.parametrize('use_causal_mask', [True, False])
    def test_keras_save_load_round_trip_preserves_values(self, use_causal_mask):
        """Save/load must both SUCCEED and restore the same numbers.

        Asserting values, not just shapes: a round-trip that silently fails to
        restore weights still produces a correctly-shaped output, so a
        shape-only assertion cannot tell a working round-trip from a broken one.
        """
        tokens = self._tokens()
        inputs = keras.Input(shape=(self.SEQ,), dtype='int32')
        model = keras.Model(inputs, self._encoder(use_causal_mask)(inputs))
        before = model.predict(tokens, verbose=0)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'text_encoder.keras')
            model.save(path)
            restored = keras.models.load_model(path)

        np.testing.assert_allclose(
            before, restored.predict(tokens, verbose=0), rtol=1e-6, atol=1e-6
        )

    @pytest.mark.parametrize('use_causal_mask', [True, False])
    def test_jit_compiled(self, use_causal_mask):
        encoder = self._encoder(use_causal_mask)

        @tf.function(jit_compile=True)
        def run(tokens):
            return encoder(tokens)

        out = run(tf.constant(self._tokens()))
        assert tuple(out.shape) == (self.BATCH, self.ENC_KWARGS['projection_dim'])

    def test_causal_mask_actually_masks_the_future(self):
        """The mask's SEMANTICS, not merely its graph-safety.

        A graph-safe mask with inverted polarity or a dropped diagonal would
        pass every test above while letting the model attend to the future.
        Two checks: the mask content matches a lower-triangular keep mask
        exactly, and enabling the mask actually changes the encoder's output
        (so `use_causal_mask=True` is not silently a no-op).
        """
        from dl_techniques.utils.masking import MaskFactory

        seq_len = 5
        mask = keras.ops.convert_to_numpy(
            keras.ops.cast(
                keras.ops.logical_not(
                    MaskFactory.create_causal_mask(seq_len, dtype='bool')
                ),
                keras.backend.floatx(),
            )
        )
        assert np.array_equal(
            mask, np.tril(np.ones((seq_len, seq_len), dtype=mask.dtype))
        )
        assert mask[0, 0] == 1.0, 'diagonal dropped: a token cannot attend to itself'
        assert mask[0, -1] == 0.0, 'future not masked'

        tokens = self._tokens()
        keras.utils.set_random_seed(99)
        with_mask = keras.ops.convert_to_numpy(
            self._encoder(True)(keras.ops.convert_to_tensor(tokens))
        )
        keras.utils.set_random_seed(99)
        without_mask = keras.ops.convert_to_numpy(
            self._encoder(False)(keras.ops.convert_to_tensor(tokens))
        )
        assert not np.allclose(with_mask, without_mask), (
            'use_causal_mask=True produced the same output as False — the mask '
            'is not reaching attention at all'
        )

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
