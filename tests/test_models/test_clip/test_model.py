"""
Comprehensive pytest test suite for CLIP (Contrastive Language-Image Pre-training) model.

This module provides extensive testing for the CLIP implementation including:
- Model initialization and parameter validation
- Architecture building and shape consistency
- Forward pass functionality for vision and text encoding
- Individual encoding operations (encode_image, encode_text)
- Serialization and deserialization
- Error handling and edge cases
- Factory function testing
- Integration testing

Refined to match the actual integrated CLIPModel implementation that uses:
- Direct configuration parameters in __init__ (no separate config class)
- Integrated vision and text processing within CLIPModel
- Modern transformer architecture with TransformerLayer components
"""

import pytest
import numpy as np
import keras
from keras import ops
import tensorflow as tf
import tempfile
import os

from dl_techniques.models.clip.model import (
    CLIPModel,
    create_clip_model,
    create_clip_variant
)


class TestCLIPModelInitialization:
    """Test CLIP model initialization and parameter validation."""

    def test_basic_initialization(self):
        """Test basic CLIPModel initialization with default parameters."""
        model = CLIPModel()

        # Check default values
        assert model.image_size == 224
        assert model.patch_size == 16
        assert model.vision_layers == 12
        assert model.vision_width == 768
        assert model.vision_heads == 12
        assert model.vision_kv_heads == 4

        assert model.vocab_size == 49408
        assert model.context_length == 77
        assert model.text_layers == 12
        assert model.text_width == 512
        assert model.text_heads == 8
        assert model.text_kv_heads == 8

        assert model.embed_dim == 512
        assert model.dropout_rate == 0.0
        assert model.attention_dropout_rate == 0.0

        # Check derived properties
        assert model.num_patches == (224 // 16) ** 2  # 196
        assert model.vision_seq_len == model.num_patches + 1  # 197 (+ CLS token)

        # Model should not be built yet
        assert model.built is False

    def test_custom_initialization(self):
        """Test CLIPModel initialization with custom parameters."""
        model = CLIPModel(
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
            text_heads=12,  # Fixed: 768/12 = 64
            text_kv_heads=12,  # Fixed: must equal text_heads for standard attention
            embed_dim=768,
            dropout_rate=0.1,
            attention_dropout_rate=0.05
        )

        assert model.image_size == 384
        assert model.patch_size == 32
        assert model.vision_layers == 24
        assert model.vision_width == 1024
        assert model.vision_heads == 16
        assert model.vision_kv_heads == 8
        assert model.vocab_size == 32000
        assert model.context_length == 128
        assert model.text_layers == 18
        assert model.text_width == 768
        assert model.text_heads == 12
        assert model.text_kv_heads == 12
        assert model.embed_dim == 768
        assert model.dropout_rate == 0.1
        assert model.attention_dropout_rate == 0.05

        # Test derived properties
        assert model.num_patches == (384 // 32) ** 2  # 144
        assert model.vision_seq_len == 144 + 1  # 145

    def test_parameter_validation(self):
        """Test CLIPModel parameter validation."""
        # Test that invalid configurations raise errors during initialization

        # Test invalid image size
        with pytest.raises(ValueError, match="image_size must be positive"):
            CLIPModel(image_size=0)

        # Test invalid patch size
        with pytest.raises(ValueError, match="patch_size must be positive"):
            CLIPModel(patch_size=-16)

        # Test non-divisible image size and patch size
        with pytest.raises(ValueError, match="image_size.*must be divisible by.*patch_size"):
            CLIPModel(image_size=224, patch_size=15)

        # Test invalid vision heads configuration
        with pytest.raises(ValueError, match="vision_width.*must be divisible by.*vision_heads"):
            CLIPModel(vision_width=768, vision_heads=11)

        # Test invalid vision GQA configuration
        with pytest.raises(ValueError, match="vision_heads.*must be divisible by.*vision_kv_heads"):
            CLIPModel(vision_heads=12, vision_kv_heads=5)

        # Test invalid text heads configuration
        with pytest.raises(ValueError, match="text_width.*must be divisible by.*text_heads"):
            CLIPModel(text_width=512, text_heads=7)

        # Test invalid text GQA configuration
        with pytest.raises(ValueError, match="text_heads.*must be divisible by.*text_kv_heads"):
            CLIPModel(text_heads=8, text_kv_heads=3)

        # Test invalid dropout rates
        with pytest.raises(ValueError, match="dropout_rate must be in"):
            CLIPModel(dropout_rate=1.5)

        with pytest.raises(ValueError, match="attention_dropout_rate must be in"):
            CLIPModel(attention_dropout_rate=-0.1)


class TestCLIPModelBuilding:
    """Test CLIP model building and architecture creation."""

    @pytest.fixture
    def basic_config_params(self) -> dict:
        """Create basic CLIP configuration parameters for testing."""
        return {
            'image_size': 224,
            'patch_size': 16,
            'vision_layers': 4,  # Smaller for testing
            'vision_width': 256,
            'vision_heads': 8,  # Fixed: 256/8 = 32
            'vision_kv_heads': 4,  # Fixed: 8/4 = 2
            'vocab_size': 1000,  # Smaller vocab
            'context_length': 32,  # Shorter context
            'text_layers': 3,  # Fewer layers
            'text_width': 256,
            'text_heads': 8,  # Fixed: 256/8 = 32
            'text_kv_heads': 4,  # Fixed: 8/4 = 2
            'embed_dim': 256
        }

    def test_build_with_dict_input_shape(self, basic_config_params):
        """Test building CLIP model with dictionary input shape."""
        model = CLIPModel(**basic_config_params)
        input_shape = {
            'image': (None, 224, 224, 3),
            'text': (None, 32)
        }

        model.build(input_shape)

        assert model.built is True
        assert model._build_input_shape == input_shape
        assert model.patch_conv is not None
        assert model.class_token is not None
        assert model.logit_scale is not None
        assert len(model.vision_transformer_layers) == 4
        assert len(model.text_transformer_layers) == 3

    def test_build_with_tuple_input_shape(self, basic_config_params):
        """Test building CLIP model with tuple input shape."""
        model = CLIPModel(**basic_config_params)
        input_shape = (
            (None, 224, 224, 3),  # Image shape
            (None, 32)  # Text shape
        )

        model.build(input_shape)

        assert model.built is True

    def test_component_shapes_after_building(self, basic_config_params):
        """Test that components have correct shapes after building."""
        model = CLIPModel(**basic_config_params)
        model.build({
            'image': (None, 224, 224, 3),
            'text': (None, 32)
        })

        # Test vision projection shape
        test_vision_features = keras.random.normal((4, basic_config_params['vision_width']))
        vision_proj_output = model.vision_projection(test_vision_features)
        assert vision_proj_output.shape == (4, basic_config_params['embed_dim'])

        # Test text projection shape
        test_text_features = keras.random.normal((4, basic_config_params['text_width']))
        text_proj_output = model.text_projection(test_text_features)
        assert text_proj_output.shape == (4, basic_config_params['embed_dim'])


class TestCLIPModelForwardPass:
    """Test CLIP model forward pass functionality."""

    @pytest.fixture
    def built_model(self) -> CLIPModel:
        """Create a built CLIP model for testing."""
        model = CLIPModel(
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
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(norms),
            1.0,
            rtol=1e-6, atol=1e-6,
            err_msg="Image features should be L2 normalized"
        )

    def test_encode_text_shapes(self, built_model):
        """Test text encoding produces correct shapes."""
        batch_size = 4
        seq_length = 32
        # Use keras.random.uniform and cast to int32
        test_tokens = ops.cast(
            keras.random.uniform(
                (batch_size, seq_length),
                minval=1,
                maxval=built_model.vocab_size
            ),
            dtype='int32'
        )

        text_features = built_model.encode_text(test_tokens, training=False)

        assert text_features.shape == (batch_size, built_model.embed_dim)

        # Features should be L2 normalized
        norms = ops.norm(text_features, axis=-1)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(norms),
            1.0,
            rtol=1e-6, atol=1e-6,
            err_msg="Text features should be L2 normalized"
        )

    def test_forward_pass_with_dict_input(self, built_model):
        """Test forward pass with dictionary input."""
        batch_size = 4
        test_images = keras.random.normal((batch_size, 224, 224, 3))
        test_tokens = ops.cast(
            keras.random.uniform(
                (batch_size, 32),
                minval=1,
                maxval=built_model.vocab_size
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
        test_tokens = ops.cast(
            keras.random.uniform(
                (batch_size, 32),
                minval=1,
                maxval=built_model.vocab_size
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
        test_tokens = ops.cast(
            keras.random.uniform(
                (batch_size, 32),
                minval=1,
                maxval=built_model.vocab_size
            ),
            dtype='int32'
        )

        inputs = {'text': test_tokens}
        outputs = built_model(inputs, training=False)

        assert 'text_features' in outputs
        assert 'image_features' not in outputs
        assert 'logits_per_image' not in outputs

    def test_similarity_matrix_properties(self, built_model):
        """Test properties of similarity matrices."""
        batch_size = 3
        test_images = keras.random.normal((batch_size, 224, 224, 3))
        test_tokens = ops.cast(
            keras.random.uniform(
                (batch_size, 32),
                minval=1,
                maxval=built_model.vocab_size
            ),
            dtype='int32'
        )

        inputs = {'image': test_images, 'text': test_tokens}
        outputs = built_model(inputs, training=False)

        logits_per_image = outputs['logits_per_image']
        logits_per_text = outputs['logits_per_text']

        # logits_per_text should be transpose of logits_per_image
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(logits_per_text),
            keras.ops.convert_to_numpy(ops.transpose(logits_per_image)),
            rtol=1e-6, atol=1e-6,
            err_msg="logits_per_text should be transpose of logits_per_image"
        )

        # Check shapes
        assert logits_per_image.shape == (batch_size, batch_size)
        assert logits_per_text.shape == (batch_size, batch_size)


class TestCLIPModelSerialization:
    """Test CLIP model serialization and deserialization."""

    def test_config_serialization(self):
        """Test model configuration serialization."""
        model = CLIPModel(
            image_size=256,
            vision_layers=8,
            text_layers=6,
            embed_dim=512,
            dropout_rate=0.1
        )

        model_config = model.get_config()

        assert isinstance(model_config, dict)
        # Check that all configuration parameters are saved
        assert model_config['image_size'] == 256
        assert model_config['vision_layers'] == 8
        assert model_config['text_layers'] == 6
        assert model_config['embed_dim'] == 512
        assert model_config['dropout_rate'] == 0.1

    def test_from_config_reconstruction(self):
        """Test model reconstruction from configuration."""
        original_model = CLIPModel(
            image_size=256,
            vision_layers=4,
            text_layers=3,
            embed_dim=256
        )

        # Get config and reconstruct
        model_config = original_model.get_config()
        reconstructed_model = CLIPModel.from_config(model_config)

        # Check that configs match
        assert reconstructed_model.image_size == original_model.image_size
        assert reconstructed_model.vision_layers == original_model.vision_layers
        assert reconstructed_model.text_layers == original_model.text_layers
        assert reconstructed_model.embed_dim == original_model.embed_dim

    def test_model_save_load(self):
        """Test saving and loading complete model."""
        # Create and build model
        model = CLIPModel(
            vision_layers=2,  # Small for testing
            text_layers=2,
            embed_dim=128
        )
        model.build({
            'image': (None, 224, 224, 3),
            'text': (None, 32)
        })

        # Test forward pass before saving
        test_images = keras.random.normal((2, 224, 224, 3))
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

            # Load with custom objects - CLIPModel should be registered
            loaded_model = keras.models.load_model(model_path)

            # Test that loaded model produces same output
            loaded_outputs = loaded_model(inputs, training=False)

            # Outputs should be very close
            for key in original_outputs.keys():
                np.testing.assert_allclose(
                    keras.ops.convert_to_numpy(original_outputs[key]),
                    keras.ops.convert_to_numpy(loaded_outputs[key]),
                    rtol=1e-5, atol=1e-6,
                    err_msg=f"Output {key} should match after loading"
                )


class TestCLIPEdgeCases:
    """Test CLIP model edge cases and error handling."""

    def test_small_input_sizes(self):
        """Test CLIP with small input sizes."""
        # Test minimum viable image size
        model = CLIPModel(
            image_size=32,  # Small image
            patch_size=16,
            vision_layers=2,
            text_layers=2,
            embed_dim=128
        )
        model.build({
            'image': (None, 32, 32, 3),
            'text': (None, 16)
        })

        test_images = keras.random.normal((2, 32, 32, 3))
        test_tokens = ops.cast(
            keras.random.uniform((2, 16), minval=1, maxval=1000),
            dtype='int32'
        )

        outputs = model({'image': test_images, 'text': test_tokens}, training=False)
        assert outputs['image_features'].shape == (2, 128)
        assert outputs['text_features'].shape == (2, 128)

    def test_single_sample_batch(self):
        """Test CLIP with batch size of 1."""
        model = CLIPModel(embed_dim=128)
        model.build({
            'image': (None, 224, 224, 3),
            'text': (None, 32)
        })

        test_images = keras.random.normal((1, 224, 224, 3))
        test_tokens = ops.cast(
            keras.random.uniform((1, 32), minval=1, maxval=1000),
            dtype='int32'
        )

        outputs = model({'image': test_images, 'text': test_tokens}, training=False)
        assert outputs['logits_per_image'].shape == (1, 1)
        assert outputs['logits_per_text'].shape == (1, 1)

    def test_text_with_heavy_padding(self):
        """Test text processing with heavy padding."""
        model = CLIPModel(embed_dim=128)
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
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(norms),
            1.0,
            rtol=1e-6, atol=1e-6,
            err_msg="Text features should be normalized even with padding"
        )


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
        assert model.image_size == 224
        assert model.vision_layers == 6
        assert model.text_layers == 4
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

        assert model.image_size == 384
        assert model.patch_size == 32
        assert model.vision_width == 576
        assert model.text_width == 512
        assert model.dropout_rate == 0.1
        assert model.attention_dropout_rate == 0.05

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
                assert model.patch_size == 32
            elif variant == "ViT-B/16":
                assert model.patch_size == 16

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
            vision_kv_heads=3,  # GQA
            text_width=384,
            text_heads=12,
            text_kv_heads=4   # Fix: ensure text_heads is divisible by text_kv_heads
        )

        model.build({
            'image': (None, 224, 224, 3),
            'text': (None, 32)
        })

        # Create test data
        batch_size = 4
        test_images = keras.random.normal((batch_size, 224, 224, 3))
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
            keras.ops.convert_to_numpy(logits_per_text),
            keras.ops.convert_to_numpy(ops.transpose(logits_per_image)),
            rtol=1e-5, atol=1e-6,
            err_msg="Similarity matrices should be transpose of each other"
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
            keras.ops.convert_to_numpy(full_outputs['image_features']),
            keras.ops.convert_to_numpy(individual_image_features),
            rtol=1e-6, atol=1e-6,
            err_msg="Individual image encoding should match full forward pass"
        )
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(full_outputs['text_features']),
            keras.ops.convert_to_numpy(individual_text_features),
            rtol=1e-6, atol=1e-6,
            err_msg="Individual text encoding should match full forward pass"
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
        grad_norms = [
            ops.sqrt(ops.sum(ops.square(g))) for g in non_none_grads
        ]
        grad_norm_values = [keras.ops.convert_to_numpy(norm) for norm in grad_norms]
        assert all(norm > 1e-12 for norm in grad_norm_values)  # Not vanishingly small
        assert all(norm < 1000.0 for norm in grad_norm_values)  # Not exploding


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])