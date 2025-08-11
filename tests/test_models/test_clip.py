"""
Comprehensive pytest test suite for CLIP (Contrastive Language-Image Pre-training) model.

This module provides extensive testing for the CLIP implementation including:
- Model initialization and parameter validation
- Architecture building and shape consistency
- Forward pass functionality for vision and text encoders
- Individual component testing (VisionTransformer, TextTransformer)
- Encoding operations and feature extraction
- Serialization and deserialization
- Error handling and edge cases
- Factory function testing
- Integration testing

Fixed issues:
- text_heads must be divisible by text_kv_heads
- vision_width must be divisible by vision_heads
- Use keras.random instead of ops.random
"""

import pytest
import numpy as np
import keras
from keras import ops
import tensorflow as tf
import tempfile
import os

from dl_techniques.models.clip import (
    CLIPConfig,
    CLIPModel,
    VisionTransformer,
    TextTransformer,
    create_clip_model,
    create_clip_variant
)


class TestCLIPConfig:
    """Test CLIP configuration validation and initialization."""

    def test_basic_initialization(self):
        """Test basic CLIPConfig initialization with default parameters."""
        config = CLIPConfig()

        # Vision encoder defaults
        assert config.image_size == 224
        assert config.patch_size == 16
        assert config.vision_layers == 12
        assert config.vision_width == 768
        assert config.vision_heads == 12
        assert config.vision_kv_heads == 4

        # Text encoder defaults
        assert config.vocab_size == 49408
        assert config.context_length == 77
        assert config.text_layers == 12
        assert config.text_width == 512
        assert config.text_heads == 8
        assert config.text_kv_heads == 8

        # Shared defaults
        assert config.embed_dim == 512

        # Derived properties
        assert config.num_patches == (224 // 16) ** 2  # 196
        assert config.vision_seq_len == config.num_patches + 1  # 197 (+ CLS token)

    def test_custom_initialization(self):
        """Test CLIPConfig initialization with custom parameters."""
        config = CLIPConfig(
            image_size=384,
            patch_size=32,
            vision_layers=24,
            vision_width=1024,
            vision_heads=16,
            vision_kv_heads=8,
            vocab_size=32000,
            context_length=128,
            text_layers=18,
            text_width=768,
            text_heads=16,
            text_kv_heads=8,  # Fixed: must be divisible by text_heads
            embed_dim=768,
            dropout_rate=0.1,
        )

        assert config.image_size == 384
        assert config.patch_size == 32
        assert config.vision_layers == 24
        assert config.vision_width == 1024
        assert config.vision_heads == 16
        assert config.vision_kv_heads == 8
        assert config.vocab_size == 32000
        assert config.context_length == 128
        assert config.text_layers == 18
        assert config.text_width == 768
        assert config.text_heads == 16
        assert config.text_kv_heads == 8
        assert config.embed_dim == 768
        assert config.dropout_rate == 0.1

        # Test derived properties
        assert config.num_patches == (384 // 32) ** 2  # 144
        assert config.vision_seq_len == 144 + 1  # 145

    def test_invalid_parameters(self):
        """Test CLIPConfig initialization with invalid parameters raises errors."""
        # Test invalid image size
        with pytest.raises(ValueError, match="image_size must be positive"):
            CLIPConfig(image_size=0)

        # Test invalid patch size
        with pytest.raises(ValueError, match="patch_size must be positive"):
            CLIPConfig(patch_size=-16)

        # Test non-divisible image size and patch size
        with pytest.raises(ValueError, match="image_size.*must be divisible by.*patch_size"):
            CLIPConfig(image_size=224, patch_size=15)

        # Test invalid vision heads configuration
        with pytest.raises(ValueError, match="vision_width.*must be divisible by.*vision_heads"):
            CLIPConfig(vision_width=768, vision_heads=11)

        # Test invalid vision GQA configuration
        with pytest.raises(ValueError, match="vision_heads.*must be divisible by.*vision_kv_heads"):
            CLIPConfig(vision_heads=12, vision_kv_heads=5)

        # Test invalid text heads configuration
        with pytest.raises(ValueError, match="text_width.*must be divisible by.*text_heads"):
            CLIPConfig(text_width=512, text_heads=7)

        # Test invalid text GQA configuration
        with pytest.raises(ValueError, match="text_heads.*must be divisible by.*text_kv_heads"):
            CLIPConfig(text_heads=8, text_kv_heads=3)

        # Test invalid dropout rates
        with pytest.raises(ValueError, match="dropout_rate must be in"):
            CLIPConfig(dropout_rate=1.5)

        with pytest.raises(ValueError, match="attention_dropout_rate must be in"):
            CLIPConfig(attention_dropout_rate=-0.1)

        # Test invalid epsilon
        with pytest.raises(ValueError, match="rms_norm_eps must be positive"):
            CLIPConfig(rms_norm_eps=-1e-6)

        # Test invalid rope percentage
        with pytest.raises(ValueError, match="rope_percentage must be in"):
            CLIPConfig(rope_percentage=1.5)

    def test_config_serialization(self):
        """Test CLIPConfig to_dict and from_dict methods."""
        original_config = CLIPConfig(
            image_size=256,
            vision_layers=8,
            text_layers=6,
            embed_dim=256
        )

        config_dict = original_config.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict['image_size'] == 256
        assert config_dict['vision_layers'] == 8
        assert config_dict['text_layers'] == 6
        assert config_dict['embed_dim'] == 256

        restored_config = CLIPConfig.from_dict(config_dict)
        assert restored_config.image_size == original_config.image_size
        assert restored_config.vision_layers == original_config.vision_layers
        assert restored_config.text_layers == original_config.text_layers
        assert restored_config.embed_dim == original_config.embed_dim


class TestVisionTransformerComponent:
    """Test VisionTransformer component individually."""

    @pytest.fixture
    def basic_config(self) -> CLIPConfig:
        """Create a basic CLIP config for testing."""
        return CLIPConfig(
            image_size=224,
            patch_size=16,
            vision_layers=6,  # Smaller for faster testing
            vision_width=384,
            vision_heads=6,  # Fixed: 384/6 = 64
            vision_kv_heads=2,
            embed_dim=256
        )

    def test_vision_transformer_initialization(self, basic_config):
        """Test VisionTransformer initialization."""
        vit = VisionTransformer(basic_config)

        assert vit.config == basic_config
        assert vit.image_size == 224
        assert vit.patch_size == 16
        assert vit.vision_width == 384
        assert vit.vision_layers == 6
        assert vit.vision_heads == 6
        assert vit.vision_kv_heads == 2
        assert vit.num_patches == (224 // 16) ** 2
        assert vit.seq_len == vit.num_patches + 1

    def test_vision_transformer_build(self, basic_config):
        """Test VisionTransformer build process."""
        vit = VisionTransformer(basic_config)
        input_shape = (None, 224, 224, 3)

        vit.build(input_shape)

        assert vit.built is True
        assert vit._build_input_shape == input_shape
        assert vit.patch_conv is not None
        assert vit.class_token is not None
        assert len(vit.transformer_layers) == 6

    def test_vision_transformer_forward_pass(self, basic_config):
        """Test VisionTransformer forward pass."""
        vit = VisionTransformer(basic_config)

        batch_size = 4
        test_images = keras.random.normal((batch_size, 224, 224, 3))

        features = vit(test_images, training=False)

        assert features.shape == (batch_size, basic_config.vision_width)

    def test_vision_transformer_different_input_sizes(self, basic_config):
        """Test VisionTransformer with different input sizes."""
        input_sizes = [
            (224, 224, 3),
            (256, 256, 3),
            (384, 384, 3),
        ]

        for height, width, channels in input_sizes:
            # Adjust config for different image sizes
            config = CLIPConfig(
                image_size=height,  # Assuming square images
                patch_size=basic_config.patch_size,
                vision_layers=3,  # Smaller for testing
                vision_width=384,  # Fixed: 384 is divisible by 6
                vision_heads=6,  # Fixed: 384/6 = 64
                vision_kv_heads=2
            )

            vit = VisionTransformer(config)
            test_input = keras.random.normal((2, height, width, channels))

            features = vit(test_input, training=False)
            assert features.shape == (2, config.vision_width)

    def test_vision_transformer_training_mode(self, basic_config):
        """Test VisionTransformer in training vs evaluation mode."""
        vit = VisionTransformer(basic_config)
        test_input = keras.random.normal((2, 224, 224, 3))

        # Test both training modes
        features_train = vit(test_input, training=True)
        features_eval = vit(test_input, training=False)

        assert features_train.shape == features_eval.shape
        assert features_train.shape == (2, basic_config.vision_width)


class TestTextTransformerComponent:
    """Test TextTransformer component individually."""

    @pytest.fixture
    def basic_config(self) -> CLIPConfig:
        """Create a basic CLIP config for testing."""
        return CLIPConfig(
            vocab_size=1000,  # Smaller vocab for testing
            context_length=32,  # Shorter context for testing
            text_layers=4,  # Fewer layers for testing
            text_width=256,
            text_heads=8,  # Fixed: 256/8 = 32
            text_kv_heads=4,  # Fixed: 8/4 = 2
            embed_dim=256
        )

    def test_text_transformer_initialization(self, basic_config):
        """Test TextTransformer initialization."""
        text_transformer = TextTransformer(basic_config)

        assert text_transformer.config == basic_config
        assert text_transformer.vocab_size == 1000
        assert text_transformer.context_length == 32
        assert text_transformer.text_width == 256
        assert text_transformer.text_layers == 4
        assert text_transformer.text_heads == 8
        assert text_transformer.text_kv_heads == 4

    def test_text_transformer_build(self, basic_config):
        """Test TextTransformer build process."""
        text_transformer = TextTransformer(basic_config)
        input_shape = (None, 32)  # (batch_size, sequence_length)

        text_transformer.build(input_shape)

        assert text_transformer.built is True
        assert text_transformer._build_input_shape == input_shape
        assert text_transformer.token_embedding is not None
        assert len(text_transformer.transformer_layers) == 4

    def test_text_transformer_forward_pass(self, basic_config):
        """Test TextTransformer forward pass."""
        text_transformer = TextTransformer(basic_config)

        batch_size = 4
        seq_length = 32
        # Create test tokens (0 is padding, so use 1 to vocab_size-1)
        # Fixed: Use keras.random.uniform and cast
        test_tokens = ops.cast(
            keras.random.uniform(
                (batch_size, seq_length),
                minval=1,
                maxval=basic_config.vocab_size
            ),
            dtype='int32'
        )

        features = text_transformer(test_tokens, training=False)

        assert features.shape == (batch_size, basic_config.text_width)

    def test_text_transformer_sequence_length_handling(self, basic_config):
        """Test TextTransformer with different sequence lengths."""
        text_transformer = TextTransformer(basic_config)
        batch_size = 3

        # Test with padding (0s at the end)
        test_tokens = ops.ones((batch_size, 32), dtype='int32')
        # Add some padding at the end for different sequence lengths
        test_tokens = ops.concatenate([
            test_tokens[:, :20],  # First 20 tokens are non-zero
            ops.zeros((batch_size, 12), dtype='int32')  # Last 12 are padding
        ], axis=1)

        features = text_transformer(test_tokens, training=False)
        assert features.shape == (batch_size, basic_config.text_width)

    def test_text_transformer_training_mode(self, basic_config):
        """Test TextTransformer in training vs evaluation mode."""
        text_transformer = TextTransformer(basic_config)
        # Fixed: Use keras.random.uniform and cast
        test_tokens = ops.cast(
            keras.random.uniform(
                (2, 32),
                minval=1,
                maxval=basic_config.vocab_size
            ),
            dtype='int32'
        )

        # Test both training modes
        features_train = text_transformer(test_tokens, training=True)
        features_eval = text_transformer(test_tokens, training=False)

        assert features_train.shape == features_eval.shape
        assert features_train.shape == (2, basic_config.text_width)


class TestCLIPModelInitialization:
    """Test CLIP model initialization and parameter validation."""

    def test_basic_initialization(self):
        """Test basic CLIP model initialization."""
        config = CLIPConfig(embed_dim=256)
        model = CLIPModel(config)

        assert model.config == config
        assert model.embed_dim == 256
        assert model.built is False

        # Components should be None before building
        assert model.vision_encoder is None
        assert model.text_encoder is None
        assert model.vision_projection is None
        assert model.text_projection is None
        assert model.logit_scale is None

    def test_initialization_with_custom_config(self):
        """Test CLIP model initialization with custom configuration."""
        config = CLIPConfig(
            image_size=256,
            vision_layers=8,
            text_layers=6,
            embed_dim=512,
            dropout_rate=0.1
        )
        model = CLIPModel(config)

        assert model.config.image_size == 256
        assert model.config.vision_layers == 8
        assert model.config.text_layers == 6
        assert model.embed_dim == 512
        assert model.config.dropout_rate == 0.1


class TestCLIPModelBuilding:
    """Test CLIP model building and architecture creation."""

    @pytest.fixture
    def basic_config(self) -> CLIPConfig:
        """Create a basic CLIP config for testing."""
        return CLIPConfig(
            image_size=224,
            patch_size=16,
            vision_layers=4,  # Smaller for testing
            vision_width=256,
            vision_heads=8,  # Fixed: 256/8 = 32
            vision_kv_heads=4,  # Fixed: 8/4 = 2
            vocab_size=1000,  # Smaller vocab
            context_length=32,  # Shorter context
            text_layers=3,  # Fewer layers
            text_width=256,
            text_heads=8,  # Fixed: 256/8 = 32
            text_kv_heads=4,  # Fixed: 8/4 = 2
            embed_dim=256
        )

    def test_build_with_dict_input_shape(self, basic_config):
        """Test building CLIP model with dictionary input shape."""
        model = CLIPModel(basic_config)
        input_shape = {
            'image': (None, 224, 224, 3),
            'text': (None, 32)
        }

        model.build(input_shape)

        assert model.built is True
        assert model._build_input_shape == input_shape
        assert model.vision_encoder is not None
        assert model.text_encoder is not None
        assert model.vision_projection is not None
        assert model.text_projection is not None
        assert model.logit_scale is not None

    def test_build_with_tuple_input_shape(self, basic_config):
        """Test building CLIP model with tuple input shape."""
        model = CLIPModel(basic_config)
        input_shape = (
            (None, 224, 224, 3),  # Image shape
            (None, 32)  # Text shape
        )

        model.build(input_shape)

        assert model.built is True
        assert model.vision_encoder.built is True
        assert model.text_encoder.built is True

    def test_build_prevents_double_building(self, basic_config):
        """Test that building twice doesn't cause issues."""
        model = CLIPModel(basic_config)
        input_shape = {
            'image': (None, 224, 224, 3),
            'text': (None, 32)
        }

        # Build first time
        model.build(input_shape)
        vision_encoder_first = model.vision_encoder
        text_encoder_first = model.text_encoder

        # Build second time
        model.build(input_shape)

        # Should be the same objects (no rebuilding)
        assert model.vision_encoder is vision_encoder_first
        assert model.text_encoder is text_encoder_first

    def test_component_shapes_after_building(self, basic_config):
        """Test that components have correct shapes after building."""
        model = CLIPModel(basic_config)
        model.build({
            'image': (None, 224, 224, 3),
            'text': (None, 32)
        })

        # Test vision projection shape
        test_vision_features = keras.random.normal((4, basic_config.vision_width))
        vision_proj_output = model.vision_projection(test_vision_features)
        assert vision_proj_output.shape == (4, basic_config.embed_dim)

        # Test text projection shape
        test_text_features = keras.random.normal((4, basic_config.text_width))
        text_proj_output = model.text_projection(test_text_features)
        assert text_proj_output.shape == (4, basic_config.embed_dim)


class TestCLIPModelForwardPass:
    """Test CLIP model forward pass functionality."""

    @pytest.fixture
    def built_model(self) -> CLIPModel:
        """Create a built CLIP model for testing."""
        config = CLIPConfig(
            image_size=224,
            patch_size=16,
            vision_layers=3,
            vision_width=256,
            vision_heads=8,  # Fixed: 256/8 = 32
            vision_kv_heads=4,  # Fixed: 8/4 = 2
            vocab_size=1000,
            context_length=32,
            text_layers=3,
            text_width=256,
            text_heads=8,  # Fixed: 256/8 = 32
            text_kv_heads=4,  # Fixed: 8/4 = 2
            embed_dim=256
        )
        model = CLIPModel(config)
        model.build({
            'image': (None, 224, 224, 3),
            'text': (None, 32)
        })
        return model

    def test_encode_image_shapes(self, built_model):
        """Test image encoding produces correct shapes."""
        batch_size = 4
        test_images = keras.random.normal((batch_size, 224, 224, 3))

        image_features = built_model.encode_image(test_images, training=False)

        assert image_features.shape == (batch_size, built_model.embed_dim)

        # Features should be L2 normalized
        norms = ops.norm(image_features, axis=-1)
        np.testing.assert_allclose(norms.numpy(), 1.0, atol=1e-6)

    def test_encode_text_shapes(self, built_model):
        """Test text encoding produces correct shapes."""
        batch_size = 4
        seq_length = 32
        # Fixed: Use keras.random.uniform and cast
        test_tokens = ops.cast(
            keras.random.uniform(
                (batch_size, seq_length),
                minval=1,
                maxval=built_model.config.vocab_size
            ),
            dtype='int32'
        )

        text_features = built_model.encode_text(test_tokens, training=False)

        assert text_features.shape == (batch_size, built_model.embed_dim)

        # Features should be L2 normalized
        norms = ops.norm(text_features, axis=-1)
        np.testing.assert_allclose(norms.numpy(), 1.0, atol=1e-6)

    def test_forward_pass_with_dict_input(self, built_model):
        """Test forward pass with dictionary input."""
        batch_size = 4
        test_images = keras.random.normal((batch_size, 224, 224, 3))
        # Fixed: Use keras.random.uniform and cast
        test_tokens = ops.cast(
            keras.random.uniform(
                (batch_size, 32),
                minval=1,
                maxval=built_model.config.vocab_size
            ),
            dtype='int32'
        )

        inputs = {'image': test_images, 'text': test_tokens}
        outputs = built_model(inputs, training=False)

        assert isinstance(outputs, dict)
        assert 'image_features' in outputs
        assert 'text_features' in outputs
        assert 'logits_per_image' in outputs
        assert 'logits_per_text' in outputs
        assert 'logit_scale' in outputs

        # Check shapes
        assert outputs['image_features'].shape == (batch_size, built_model.embed_dim)
        assert outputs['text_features'].shape == (batch_size, built_model.embed_dim)
        assert outputs['logits_per_image'].shape == (batch_size, batch_size)
        assert outputs['logits_per_text'].shape == (batch_size, batch_size)
        assert outputs['logit_scale'].shape == ()

    def test_forward_pass_with_tuple_input(self, built_model):
        """Test forward pass with tuple input."""
        batch_size = 4
        test_images = keras.random.normal((batch_size, 224, 224, 3))
        # Fixed: Use keras.random.uniform and cast
        test_tokens = ops.cast(
            keras.random.uniform(
                (batch_size, 32),
                minval=1,
                maxval=built_model.config.vocab_size
            ),
            dtype='int32'
        )

        inputs = (test_images, test_tokens)
        outputs = built_model(inputs, training=False)

        assert isinstance(outputs, dict)
        assert 'logits_per_image' in outputs
        assert 'logits_per_text' in outputs

    def test_forward_pass_image_only(self, built_model):
        """Test forward pass with only images."""
        batch_size = 4
        test_images = keras.random.normal((batch_size, 224, 224, 3))

        inputs = {'image': test_images}
        outputs = built_model(inputs, training=False)

        assert 'image_features' in outputs
        assert 'text_features' not in outputs
        assert 'logits_per_image' not in outputs

    def test_forward_pass_text_only(self, built_model):
        """Test forward pass with only text."""
        batch_size = 4
        # Fixed: Use keras.random.uniform and cast
        test_tokens = ops.cast(
            keras.random.uniform(
                (batch_size, 32),
                minval=1,
                maxval=built_model.config.vocab_size
            ),
            dtype='int32'
        )

        inputs = {'text': test_tokens}
        outputs = built_model(inputs, training=False)

        assert 'text_features' in outputs
        assert 'image_features' not in outputs
        assert 'logits_per_image' not in outputs

    def test_forward_pass_training_mode(self, built_model):
        """Test forward pass in training vs evaluation mode."""
        batch_size = 2
        test_images = keras.random.normal((batch_size, 224, 224, 3))
        # Fixed: Use keras.random.uniform and cast
        test_tokens = ops.cast(
            keras.random.uniform(
                (batch_size, 32),
                minval=1,
                maxval=built_model.config.vocab_size
            ),
            dtype='int32'
        )

        inputs = {'image': test_images, 'text': test_tokens}

        # Test both modes
        outputs_train = built_model(inputs, training=True)
        outputs_eval = built_model(inputs, training=False)

        # Both should have same structure and shapes
        assert set(outputs_train.keys()) == set(outputs_eval.keys())
        for key in outputs_train.keys():
            assert outputs_train[key].shape == outputs_eval[key].shape

    def test_similarity_matrix_properties(self, built_model):
        """Test properties of similarity matrices."""
        batch_size = 3
        test_images = keras.random.normal((batch_size, 224, 224, 3))
        # Fixed: Use keras.random.uniform and cast
        test_tokens = ops.cast(
            keras.random.uniform(
                (batch_size, 32),
                minval=1,
                maxval=built_model.config.vocab_size
            ),
            dtype='int32'
        )

        inputs = {'image': test_images, 'text': test_tokens}
        outputs = built_model(inputs, training=False)

        logits_per_image = outputs['logits_per_image']
        logits_per_text = outputs['logits_per_text']

        # logits_per_text should be transpose of logits_per_image
        np.testing.assert_allclose(
            logits_per_text.numpy(),
            ops.transpose(logits_per_image).numpy(),
            rtol=1e-5
        )

        # Diagonal elements should be highest (self-similarity)
        # This might not always be true with random data, but we test structure
        assert logits_per_image.shape == (batch_size, batch_size)
        assert logits_per_text.shape == (batch_size, batch_size)


class TestCLIPModelSerialization:
    """Test CLIP model serialization and deserialization."""

    def test_config_serialization(self):
        """Test model configuration serialization."""
        config = CLIPConfig(
            image_size=256,
            vision_layers=8,
            text_layers=6,
            embed_dim=512,
            dropout_rate=0.1
        )
        model = CLIPModel(config)

        model_config = model.get_config()

        assert isinstance(model_config, dict)
        assert 'config' in model_config

        config_dict = model_config['config']
        assert config_dict['image_size'] == 256
        assert config_dict['vision_layers'] == 8
        assert config_dict['text_layers'] == 6
        assert config_dict['embed_dim'] == 512
        assert config_dict['dropout_rate'] == 0.1

    def test_from_config_reconstruction(self):
        """Test model reconstruction from configuration."""
        original_config = CLIPConfig(
            image_size=256,
            vision_layers=4,
            text_layers=3,
            embed_dim=256
        )
        original_model = CLIPModel(original_config)

        # Get config and reconstruct
        model_config = original_model.get_config()
        reconstructed_model = CLIPModel.from_config(model_config)

        # Check that configs match
        assert reconstructed_model.config.image_size == original_config.image_size
        assert reconstructed_model.config.vision_layers == original_config.vision_layers
        assert reconstructed_model.config.text_layers == original_config.text_layers
        assert reconstructed_model.config.embed_dim == original_config.embed_dim

    def test_build_config_serialization(self):
        """Test build configuration serialization."""
        config = CLIPConfig(embed_dim=256)
        model = CLIPModel(config)

        input_shape = {
            'image': (None, 224, 224, 3),
            'text': (None, 32)
        }
        model.build(input_shape)

        build_config = model.get_build_config()
        assert build_config['input_shape'] == input_shape

        # Test build from config
        new_model = CLIPModel(config)
        new_model.build_from_config(build_config)
        assert new_model.built is True

    def test_component_serialization(self):
        """Test individual component serialization."""
        config = CLIPConfig(embed_dim=256)

        # Test VisionTransformer serialization
        vit = VisionTransformer(config)
        vit_config = vit.get_config()
        reconstructed_vit = VisionTransformer.from_config(vit_config)
        assert reconstructed_vit.config.image_size == config.image_size

        # Test TextTransformer serialization
        text_transformer = TextTransformer(config)
        text_config = text_transformer.get_config()
        reconstructed_text = TextTransformer.from_config(text_config)
        assert reconstructed_text.config.vocab_size == config.vocab_size

    def test_model_save_load(self):
        """Test saving and loading complete model."""
        # Create and build model
        config = CLIPConfig(
            vision_layers=2,  # Small for testing
            text_layers=2,
            embed_dim=128
        )
        model = CLIPModel(config)
        model.build({
            'image': (None, 224, 224, 3),
            'text': (None, 32)
        })

        # Test forward pass before saving
        test_images = keras.random.normal((2, 224, 224, 3))
        # Fixed: Use keras.random.uniform and cast
        test_tokens = ops.cast(
            keras.random.uniform((2, 32), minval=1, maxval=1000),
            dtype='int32'
        )
        inputs = {'image': test_images, 'text': test_tokens}

        original_outputs = model(inputs, training=False)

        # Save and load model
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'test_clip.keras')
            model.save(model_path)

            loaded_model = keras.models.load_model(
                model_path,
                custom_objects={
                    'CLIPModel': CLIPModel,
                    'VisionTransformer': VisionTransformer,
                    'TextTransformer': TextTransformer,
                    'CLIPConfig': CLIPConfig
                }
            )

            # Test that loaded model produces same output
            loaded_outputs = loaded_model(inputs, training=False)

            # Outputs should be very close
            for key in original_outputs.keys():
                np.testing.assert_allclose(
                    original_outputs[key].numpy(),
                    loaded_outputs[key].numpy(),
                    rtol=1e-5,
                    atol=1e-6
                )


class TestCLIPEdgeCases:
    """Test CLIP model edge cases and error handling."""

    def test_small_input_sizes(self):
        """Test CLIP with small input sizes."""
        # Test minimum viable image size
        config = CLIPConfig(
            image_size=32,  # Small image
            patch_size=16,
            vision_layers=2,
            text_layers=2,
            embed_dim=128
        )
        model = CLIPModel(config)
        model.build({
            'image': (None, 32, 32, 3),
            'text': (None, 16)
        })

        test_images = keras.random.normal((2, 32, 32, 3))
        # Fixed: Use keras.random.uniform and cast
        test_tokens = ops.cast(
            keras.random.uniform((2, 16), minval=1, maxval=1000),
            dtype='int32'
        )

        outputs = model({'image': test_images, 'text': test_tokens}, training=False)
        assert outputs['image_features'].shape == (2, 128)
        assert outputs['text_features'].shape == (2, 128)

    def test_single_sample_batch(self):
        """Test CLIP with batch size of 1."""
        config = CLIPConfig(embed_dim=128)
        model = CLIPModel(config)
        model.build({
            'image': (None, 224, 224, 3),
            'text': (None, 32)
        })

        test_images = keras.random.normal((1, 224, 224, 3))
        # Fixed: Use keras.random.uniform and cast
        test_tokens = ops.cast(
            keras.random.uniform((1, 32), minval=1, maxval=1000),
            dtype='int32'
        )

        outputs = model({'image': test_images, 'text': test_tokens}, training=False)
        assert outputs['logits_per_image'].shape == (1, 1)
        assert outputs['logits_per_text'].shape == (1, 1)

    def test_different_batch_sizes(self):
        """Test CLIP with different batch sizes."""
        config = CLIPConfig(embed_dim=128)
        model = CLIPModel(config)
        model.build({
            'image': (None, 224, 224, 3),
            'text': (None, 32)
        })

        batch_sizes = [1, 3, 8, 16]

        for batch_size in batch_sizes:
            test_images = keras.random.normal((batch_size, 224, 224, 3))
            # Fixed: Use keras.random.uniform and cast
            test_tokens = ops.cast(
                keras.random.uniform(
                    (batch_size, 32), minval=1, maxval=1000
                ),
                dtype='int32'
            )

            outputs = model({'image': test_images, 'text': test_tokens}, training=False)

            assert outputs['image_features'].shape[0] == batch_size
            assert outputs['text_features'].shape[0] == batch_size
            assert outputs['logits_per_image'].shape == (batch_size, batch_size)

    def test_grayscale_images(self):
        """Test CLIP with grayscale images."""
        config = CLIPConfig(embed_dim=128)
        model = CLIPModel(config)
        model.build({
            'image': (None, 224, 224, 1),  # Grayscale
            'text': (None, 32)
        })

        test_images = keras.random.normal((2, 224, 224, 1))
        # Fixed: Use keras.random.uniform and cast
        test_tokens = ops.cast(
            keras.random.uniform((2, 32), minval=1, maxval=1000),
            dtype='int32'
        )

        outputs = model({'image': test_images, 'text': test_tokens}, training=False)
        assert outputs['image_features'].shape == (2, 128)

    def test_very_long_text_sequences(self):
        """Test CLIP with maximum context length text."""
        config = CLIPConfig(
            context_length=77,  # Standard CLIP context length
            embed_dim=128
        )
        model = CLIPModel(config)
        model.build({
            'image': (None, 224, 224, 3),
            'text': (None, 77)
        })

        test_images = keras.random.normal((2, 224, 224, 3))
        # Fixed: Use keras.random.uniform and cast
        test_tokens = ops.cast(
            keras.random.uniform((2, 77), minval=1, maxval=1000),
            dtype='int32'
        )

        outputs = model({'image': test_images, 'text': test_tokens}, training=False)
        assert outputs['text_features'].shape == (2, 128)

    def test_text_with_heavy_padding(self):
        """Test text processing with heavy padding."""
        config = CLIPConfig(embed_dim=128)
        model = CLIPModel(config)
        model.build({
            'image': (None, 224, 224, 3),
            'text': (None, 32)
        })

        # Create text with lots of padding (0s)
        test_tokens = ops.ones((2, 32), dtype='int32')
        # Only first 5 tokens are non-zero, rest are padding
        padded_tokens = ops.concatenate([
            test_tokens[:, :5],
            ops.zeros((2, 27), dtype='int32')
        ], axis=1)

        test_images = keras.random.normal((2, 224, 224, 3))
        outputs = model({'image': test_images, 'text': padded_tokens}, training=False)

        assert outputs['text_features'].shape == (2, 128)
        # Features should still be normalized
        norms = ops.norm(outputs['text_features'], axis=-1)
        np.testing.assert_allclose(norms.numpy(), 1.0, atol=1e-6)


class TestCreateCLIPFactory:
    """Test the create_clip_model factory function."""

    def test_create_clip_model_basic(self):
        """Test basic create_clip_model functionality."""
        model = create_clip_model(
            image_size=224,
            vision_layers=6,
            text_layers=4,
            embed_dim=256
        )

        assert isinstance(model, CLIPModel)
        assert model.config.image_size == 224
        assert model.config.vision_layers == 6
        assert model.config.text_layers == 4
        assert model.embed_dim == 256

    def test_create_clip_model_with_custom_parameters(self):
        """Test create_clip_model with custom parameters."""
        model = create_clip_model(
            image_size=384,
            patch_size=32,
            vision_width=576,  # Fixed: 576/12 = 48
            vision_heads=12,
            text_width=512,  # Fixed: 512/8 = 64
            text_heads=8,
            dropout_rate=0.1,
            attention_dropout_rate=0.05
        )

        assert model.config.image_size == 384
        assert model.config.patch_size == 32
        assert model.config.vision_width == 576
        assert model.config.text_width == 512
        assert model.config.dropout_rate == 0.1
        assert model.config.attention_dropout_rate == 0.05

    def test_create_clip_model_functional(self):
        """Test that created model is functional."""
        model = create_clip_model(
            vision_layers=2,  # Small for testing
            text_layers=2,
            embed_dim=128
        )

        # Build and test
        model.build({
            'image': (None, 224, 224, 3),
            'text': (None, 32)
        })

        test_images = keras.random.normal((2, 224, 224, 3))
        # Fixed: Use keras.random.uniform and cast
        test_tokens = ops.cast(
            keras.random.uniform((2, 32), minval=1, maxval=1000),
            dtype='int32'
        )

        outputs = model({'image': test_images, 'text': test_tokens}, training=False)

        assert 'image_features' in outputs
        assert 'text_features' in outputs
        assert outputs['image_features'].shape == (2, 128)

    def test_create_clip_variant_basic(self):
        """Test create_clip_variant with predefined configurations."""
        variants = ["ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-H/14"]

        for variant in variants:
            model = create_clip_variant(variant)
            assert isinstance(model, CLIPModel)

            # Check that different variants have different configurations
            if variant == "ViT-B/32":
                assert model.config.patch_size == 32
            elif variant == "ViT-B/16":
                assert model.config.patch_size == 16

    def test_create_clip_variant_invalid(self):
        """Test create_clip_variant with invalid variant name."""
        with pytest.raises(ValueError, match="Unknown variant"):
            create_clip_variant("Invalid-Variant")

    def test_create_clip_variant_functional(self):
        """Test that created variant is functional."""
        model = create_clip_variant("ViT-B/16")

        model.build({
            'image': (None, 224, 224, 3),
            'text': (None, 77)
        })

        test_images = keras.random.normal((2, 224, 224, 3))
        # Fixed: Use keras.random.uniform and cast
        test_tokens = ops.cast(
            keras.random.uniform((2, 77), minval=1, maxval=49408),
            dtype='int32'
        )

        outputs = model({'image': test_images, 'text': test_tokens}, training=False)
        assert outputs['image_features'].shape == (2, 512)  # ViT-B/16 embed_dim


class TestCLIPIntegration:
    """Integration tests for CLIP components working together."""

    def test_end_to_end_similarity_computation(self):
        """Test complete similarity computation workflow."""
        # Create model with small architecture for testing
        model = create_clip_model(
            vision_layers=3,
            text_layers=3,
            embed_dim=256,
            vision_width=384,
            vision_heads=12,
            vision_kv_heads=3, # GQA
            text_width=384,
            text_heads=12,
            text_kv_heads=4    # Fix: ensure text_heads is divisible by text_kv_heads
        )

        model.build({
            'image': (None, 224, 224, 3),
            'text': (None, 32)
        })

        # Create test data
        batch_size = 4
        test_images = keras.random.normal((batch_size, 224, 224, 3))
        # Fixed: Use keras.random.uniform and cast
        test_tokens = ops.cast(
            keras.random.uniform(
                (batch_size, 32), minval=1, maxval=1000
            ),
            dtype='int32'
        )

        # Forward pass
        outputs = model({'image': test_images, 'text': test_tokens}, training=False)

        # Test similarity properties
        logits_per_image = outputs['logits_per_image']
        logits_per_text = outputs['logits_per_text']

        # Check symmetry
        np.testing.assert_allclose(
            logits_per_text.numpy(),
            ops.transpose(logits_per_image).numpy(),
            rtol=1e-5
        )

        # Check temperature scaling is applied
        logit_scale = outputs['logit_scale']
        assert logit_scale > 1.0  # Should be exp(logit_scale_init) > 1

    def test_component_consistency(self):
        """Test consistency between individual encoding and full forward pass."""
        model = create_clip_model(
            vision_layers=2,
            text_layers=2,
            embed_dim=128
        )

        model.build({
            'image': (None, 224, 224, 3),
            'text': (None, 32)
        })

        test_images = keras.random.normal((3, 224, 224, 3))
        # Fixed: Use keras.random.uniform and cast
        test_tokens = ops.cast(
            keras.random.uniform((3, 32), minval=1, maxval=1000),
            dtype='int32'
        )

        # Full forward pass
        full_outputs = model({'image': test_images, 'text': test_tokens}, training=False)

        # Individual encoding
        individual_image_features = model.encode_image(test_images, training=False)
        individual_text_features = model.encode_text(test_tokens, training=False)

        # Should be identical
        np.testing.assert_allclose(
            full_outputs['image_features'].numpy(),
            individual_image_features.numpy(),
            rtol=1e-6
        )
        np.testing.assert_allclose(
            full_outputs['text_features'].numpy(),
            individual_text_features.numpy(),
            rtol=1e-6
        )

    def test_feature_quality_basic(self):
        """Test basic feature quality properties."""
        model = create_clip_model(
            vision_layers=2,
            text_layers=2,
            embed_dim=128
        )

        model.build({
            'image': (None, 224, 224, 3),
            'text': (None, 32)
        })

        # Test feature normalization
        test_images = keras.random.normal((4, 224, 224, 3))
        image_features = model.encode_image(test_images, training=False)

        # Features should be unit normalized
        norms = ops.norm(image_features, axis=-1)
        np.testing.assert_allclose(norms.numpy(), 1.0, atol=1e-6)

        # Features should have reasonable variance (not all the same)
        feature_std = ops.std(image_features, axis=0)
        assert ops.mean(feature_std) > 0.01  # Some variation across dimensions

    def test_training_vs_inference_consistency(self):
        """Test consistency between training and inference modes."""
        model = create_clip_model(
            vision_layers=2,
            text_layers=2,
            embed_dim=128,
            dropout_rate=0.0,  # No dropout for consistent results
            attention_dropout_rate=0.0,
        )

        model.build({
            'image': (None, 224, 224, 3),
            'text': (None, 32)
        })

        test_images = keras.random.normal((2, 224, 224, 3))
        # Fixed: Use keras.random.uniform and cast
        test_tokens = ops.cast(
            keras.random.uniform((2, 32), minval=1, maxval=1000),
            dtype='int32'
        )

        # Compare training and inference modes (should be identical with no dropout)
        outputs_train = model({'image': test_images, 'text': test_tokens}, training=True)
        outputs_eval = model({'image': test_images, 'text': test_tokens}, training=False)

        # With no dropout/stochastic depth, results should be very close
        np.testing.assert_allclose(
            outputs_train['image_features'].numpy(),
            outputs_eval['image_features'].numpy(),
            rtol=1e-5
        )

    def test_gradient_flow_integration(self):
        """Test that gradients flow through the entire model."""
        model = create_clip_model(
            vision_layers=2,
            text_layers=2,
            embed_dim=64  # Small for faster testing
        )

        model.build({
            'image': (None, 224, 224, 3),
            'text': (None, 32)
        })

        test_images = keras.random.normal((2, 224, 224, 3))
        # Fixed: Use keras.random.uniform and cast
        test_tokens = ops.cast(
            keras.random.uniform((2, 32), minval=1, maxval=1000),
            dtype='int32'
        )

        with tf.GradientTape() as tape:
            outputs = model({'image': test_images, 'text': test_tokens}, training=True)

            # Create a simple contrastive-like loss
            logits_per_image = outputs['logits_per_image']
            # Simple loss: negative diagonal (matching pairs should have high similarity)
            loss = -ops.mean(ops.diagonal(logits_per_image))

        gradients = tape.gradient(loss, model.trainable_weights)

        # Check that we have gradients for most weights
        non_none_grads = [g for g in gradients if g is not None]
        assert len(non_none_grads) > len(model.trainable_weights) * 0.8

        # Check that gradients have reasonable magnitudes
        # FIX: ops.norm fails on tensors with rank > 2 when axis=None.
        # We compute the L2 norm manually for a robust, backend-agnostic solution.
        grad_norms = [
            ops.sqrt(ops.sum(ops.square(g))) for g in non_none_grads
        ]
        assert all(norm > 1e-12 for norm in grad_norms)  # Not vanishingly small
        assert all(norm < 1000.0 for norm in grad_norms)  # Not exploding


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])