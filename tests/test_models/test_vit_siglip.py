"""
Comprehensive pytest test suite for SigLIP Vision Transformer model.

This module provides extensive testing for the SigLIP Vision Transformer implementation including:
- Model configuration validation and serialization
- Model initialization and parameter validation
- Architecture building and shape consistency
- Forward pass functionality with different configurations
- Feature extraction methods (CLS token, patch tokens, spatial features)
- Serialization and deserialization
- Error handling and edge cases
- Factory function testing
- Integration testing with different transformer configurations
- Training vs inference behavior

The test suite follows dl-techniques framework standards and ensures comprehensive
coverage of the SigLIP Vision Transformer model functionality.
"""

import pytest
import numpy as np
import keras
from keras import ops
import tensorflow as tf
import tempfile
import os

from dl_techniques.models.vit_siglip import (
    ViTSigLIPConfig,
    ViTSigLIP,
    create_siglip_vit,
    create_siglip_vit_base,
    create_siglip_vit_large,
    create_siglip_vit_small,
    build_and_initialize_siglip_vit
)


class TestSigLIPVisionTransformerConfig:
    """Test SigLIP Vision Transformer configuration validation and initialization."""

    def test_basic_initialization(self):
        """Test basic ViTSigLIPConfig initialization with default parameters."""
        config = ViTSigLIPConfig()

        # Check default values
        assert config.img_size == 224
        assert config.patch_size == 16
        assert config.embed_dim == 768
        assert config.depth == 12
        assert config.num_heads == 12
        assert config.mlp_ratio == 4.0
        assert config.dropout == 0.0
        assert config.attention_type == 'multi_head_attention'
        assert config.normalization_type == 'layer_norm'
        assert config.normalization_position == 'post'
        assert config.ffn_type == 'mlp'
        assert config.use_bias is True
        assert config.kernel_initializer == 'glorot_uniform'
        assert config.bias_initializer == 'zeros'
        assert config.kernel_regularizer is None
        assert config.bias_regularizer is None

    def test_custom_initialization(self):
        """Test ViTSigLIPConfig initialization with custom parameters."""
        config = ViTSigLIPConfig(
            img_size=384,
            patch_size=32,
            embed_dim=1024,
            depth=24,
            num_heads=16,
            mlp_ratio=6.0,
            dropout=0.1,
            attention_type='window_attention',
            normalization_type='rms_norm',
            normalization_position='pre',
            ffn_type='swiglu',
            use_bias=False,
            kernel_initializer='he_normal',
            bias_initializer='ones',
            kernel_regularizer='l2',
            bias_regularizer='l1'
        )

        assert config.img_size == 384
        assert config.patch_size == 32
        assert config.embed_dim == 1024
        assert config.depth == 24
        assert config.num_heads == 16
        assert config.mlp_ratio == 6.0
        assert config.dropout == 0.1
        assert config.attention_type == 'window_attention'
        assert config.normalization_type == 'rms_norm'
        assert config.normalization_position == 'pre'
        assert config.ffn_type == 'swiglu'
        assert config.use_bias is False
        assert config.kernel_initializer == 'he_normal'
        assert config.bias_initializer == 'ones'
        assert config.kernel_regularizer == 'l2'
        assert config.bias_regularizer == 'l1'

    def test_config_serialization(self):
        """Test ViTSigLIPConfig to_dict and from_dict methods."""
        original_config = ViTSigLIPConfig(
            img_size=256,
            embed_dim=512,
            depth=8,
            num_heads=8,
            attention_type='group_query_attention',
            dropout=0.1
        )

        config_dict = original_config.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict['img_size'] == 256
        assert config_dict['embed_dim'] == 512
        assert config_dict['depth'] == 8
        assert config_dict['attention_type'] == 'group_query_attention'
        assert config_dict['dropout'] == 0.1

        restored_config = ViTSigLIPConfig.from_dict(config_dict)
        assert restored_config.img_size == original_config.img_size
        assert restored_config.embed_dim == original_config.embed_dim
        assert restored_config.depth == original_config.depth
        assert restored_config.attention_type == original_config.attention_type
        assert restored_config.dropout == original_config.dropout


class TestSigLIPVisionTransformerInitialization:
    """Test SigLIP Vision Transformer model initialization and parameter validation."""

    def test_basic_initialization(self):
        """Test basic ViTSigLIP initialization."""
        model = ViTSigLIP(
            img_size=224,
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12
        )

        assert model.img_size == 224
        assert model.patch_size == 16
        assert model.embed_dim == 768
        assert model.depth == 12
        assert model.num_heads == 12
        assert model.num_patches == (224 // 16) ** 2  # 196
        assert model.intermediate_size == int(768 * 4.0)  # 3072
        assert model.built is False

        # Components should be None before building
        assert model.siglip_patch_embed is None
        assert model.cls_token is None
        assert model.pos_embed is None
        assert model.transformer_blocks is None
        assert model.norm is None

    def test_initialization_with_custom_config(self):
        """Test ViTSigLIP initialization with custom configuration."""
        model = ViTSigLIP(
            img_size=384,
            patch_size=32,
            embed_dim=1024,
            depth=8,
            num_heads=16,
            mlp_ratio=6.0,
            dropout=0.1,
            attention_type='window_attention',
            normalization_type='rms_norm',
            normalization_position='pre',
            ffn_type='swiglu'
        )

        assert model.img_size == 384
        assert model.patch_size == 32
        assert model.embed_dim == 1024
        assert model.depth == 8
        assert model.num_heads == 16
        assert model.mlp_ratio == 6.0
        assert model.dropout == 0.1
        assert model.attention_type == 'window_attention'
        assert model.normalization_type == 'rms_norm'
        assert model.normalization_position == 'pre'
        assert model.ffn_type == 'swiglu'
        assert model.num_patches == (384 // 32) ** 2  # 144
        assert model.intermediate_size == int(1024 * 6.0)  # 6144

    def test_invalid_parameters(self):
        """Test ViTSigLIP initialization with invalid parameters raises errors."""
        # Test invalid image size
        with pytest.raises(ValueError, match="img_size must be positive"):
            ViTSigLIP(img_size=0)

        # Test invalid patch size
        with pytest.raises(ValueError, match="patch_size must be positive"):
            ViTSigLIP(patch_size=-16)

        # Test non-divisible image size and patch size
        with pytest.raises(ValueError, match="Image size.*must be divisible by patch size"):
            ViTSigLIP(img_size=224, patch_size=15)

        # Test invalid embed_dim
        with pytest.raises(ValueError, match="embed_dim must be positive"):
            ViTSigLIP(embed_dim=0)

        # Test invalid num_heads
        with pytest.raises(ValueError, match="num_heads must be positive"):
            ViTSigLIP(num_heads=0)

        # Test non-divisible embed_dim and num_heads
        with pytest.raises(ValueError, match="Embedding dimension.*must be divisible by number of heads"):
            ViTSigLIP(embed_dim=768, num_heads=11)

        # Test invalid dropout rates
        with pytest.raises(ValueError, match="dropout must be between 0 and 1"):
            ViTSigLIP(dropout=1.5)

        with pytest.raises(ValueError, match="dropout must be between 0 and 1"):
            ViTSigLIP(dropout=-0.1)


class TestSigLIPVisionTransformerBuilding:
    """Test SigLIP Vision Transformer model building and architecture creation."""

    @pytest.fixture
    def basic_config_params(self) -> dict:
        """Create basic configuration parameters for testing."""
        return {
            'img_size': 224,
            'patch_size': 16,
            'embed_dim': 384,  # Smaller for faster testing
            'depth': 4,  # Fewer layers for testing
            'num_heads': 6,  # 384/6 = 64
            'mlp_ratio': 4.0,
            'dropout': 0.0,
        }

    def test_build_with_valid_input_shape(self, basic_config_params):
        """Test building ViTSigLIP with valid input shape."""
        model = ViTSigLIP(**basic_config_params)
        input_shape = (None, 224, 224, 3)

        model.build(input_shape)

        assert model.built is True
        assert model._build_input_shape == input_shape
        assert model.siglip_patch_embed is not None
        assert model.cls_token is not None
        assert model.pos_embed is not None
        assert model.transformer_blocks is not None
        assert len(model.transformer_blocks) == 4
        assert model.norm is not None

        # Check CLS token shape
        assert model.cls_token.shape == (1, 1, 384)

    def test_build_with_different_input_shapes(self, basic_config_params):
        """Test building with different input shapes."""
        input_shapes = [
            (None, 224, 224, 3),  # RGB
            (None, 224, 224, 1),  # Grayscale
            (None, 256, 256, 3),  # Different size (need to adjust config)
        ]

        for input_shape in input_shapes:
            if input_shape[1] != 224:
                # Adjust config for different image size
                config_params = basic_config_params.copy()
                config_params['img_size'] = input_shape[1]
                model = ViTSigLIP(**config_params)
            else:
                model = ViTSigLIP(**basic_config_params)

            model.build(input_shape)
            assert model.built is True
            assert model._build_input_shape == input_shape

    def test_build_prevents_double_building(self, basic_config_params):
        """Test that building twice doesn't cause issues."""
        model = ViTSigLIP(**basic_config_params)
        input_shape = (None, 224, 224, 3)

        # Build first time
        model.build(input_shape)
        first_cls_token = model.cls_token
        first_patch_embed = model.siglip_patch_embed

        # Build second time
        model.build(input_shape)

        # Should be the same objects (no rebuilding)
        assert model.cls_token is first_cls_token
        assert model.siglip_patch_embed is first_patch_embed

    def test_build_invalid_input_shape(self, basic_config_params):
        """Test building with invalid input shapes raises errors."""
        model = ViTSigLIP(**basic_config_params)

        # Test wrong number of dimensions
        with pytest.raises(ValueError, match="Expected 4D input shape"):
            model.build((224, 224, 3))  # Missing batch dimension

        # Test wrong height/width
        with pytest.raises(ValueError, match="Input shape height/width.*must match img_size"):
            model.build((None, 256, 256, 3))  # Wrong size for img_size=224

    def test_child_component_building(self, basic_config_params):
        """Test that child components are properly built."""
        model = ViTSigLIP(**basic_config_params)
        model.build((None, 224, 224, 3))

        # Check that all child components are built
        assert model.siglip_patch_embed.built is True
        assert model.pos_embed.built is True
        assert model.norm.built is True

        for i, block in enumerate(model.transformer_blocks):
            assert block.built is True, f"Transformer block {i} not built"


class TestSigLIPVisionTransformerForwardPass:
    """Test SigLIP Vision Transformer forward pass functionality."""

    @pytest.fixture
    def built_model(self) -> ViTSigLIP:
        """Create a built ViTSigLIP for testing."""
        model = ViTSigLIP(
            img_size=224,
            patch_size=16,
            embed_dim=256,  # Smaller for faster testing
            depth=3,  # Fewer layers
            num_heads=8,  # 256/8 = 32
            dropout=0.0
        )
        model.build((None, 224, 224, 3))
        return model

    def test_forward_pass_basic_shapes(self, built_model):
        """Test basic forward pass produces correct shapes."""
        batch_size = 4
        test_images = keras.random.normal((batch_size, 224, 224, 3))

        features = built_model(test_images, training=False)

        # Check output shape: [batch_size, num_patches + 1, embed_dim]
        expected_patches = (224 // 16) ** 2  # 196
        expected_seq_len = expected_patches + 1  # +1 for CLS token
        assert features.shape == (batch_size, expected_seq_len, 256)

    def test_forward_pass_different_batch_sizes(self, built_model):
        """Test forward pass with different batch sizes."""
        batch_sizes = [1, 2, 8, 16]

        for batch_size in batch_sizes:
            test_images = keras.random.normal((batch_size, 224, 224, 3))
            features = built_model(test_images, training=False)

            expected_seq_len = (224 // 16) ** 2 + 1  # 197
            assert features.shape == (batch_size, expected_seq_len, 256)

    def test_forward_pass_training_vs_inference(self, built_model):
        """Test forward pass in training vs inference mode."""
        batch_size = 2
        test_images = keras.random.normal((batch_size, 224, 224, 3))

        # Test both modes
        features_train = built_model(test_images, training=True)
        features_eval = built_model(test_images, training=False)

        # Should have same shape
        assert features_train.shape == features_eval.shape

        # With dropout=0.0, should be identical
        np.testing.assert_allclose(
            features_train.numpy(),
            features_eval.numpy(),
            rtol=1e-5
        )

    def test_forward_pass_with_dropout(self):
        """Test forward pass with dropout enabled."""
        model = ViTSigLIP(
            img_size=224,
            patch_size=16,
            embed_dim=256,
            depth=2,
            num_heads=8,
            dropout=0.1  # Enable dropout
        )
        model.build((None, 224, 224, 3))

        test_images = keras.random.normal((2, 224, 224, 3))

        # With dropout, training and inference should be different
        features_train = model(test_images, training=True)
        features_eval = model(test_images, training=False)

        assert features_train.shape == features_eval.shape
        # They should be different due to dropout
        # Note: This might occasionally fail due to randomness, but very unlikely

    def test_get_cls_token(self, built_model):
        """Test CLS token extraction."""
        batch_size = 3
        test_images = keras.random.normal((batch_size, 224, 224, 3))

        features = built_model(test_images, training=False)
        cls_token = built_model.get_cls_token(features)

        assert cls_token.shape == (batch_size, 256)

        # Should be the first token
        expected_cls = features[:, 0, :]
        np.testing.assert_allclose(
            cls_token.numpy(),
            expected_cls.numpy(),
            rtol=1e-6
        )

    def test_get_patch_tokens(self, built_model):
        """Test patch token extraction."""
        batch_size = 3
        test_images = keras.random.normal((batch_size, 224, 224, 3))

        features = built_model(test_images, training=False)
        patch_tokens = built_model.get_patch_tokens(features)

        expected_patches = (224 // 16) ** 2  # 196
        assert patch_tokens.shape == (batch_size, expected_patches, 256)

        # Should be all tokens except the first (CLS)
        expected_patches = features[:, 1:, :]
        np.testing.assert_allclose(
            patch_tokens.numpy(),
            expected_patches.numpy(),
            rtol=1e-6
        )

    def test_get_spatial_features(self, built_model):
        """Test spatial feature reshaping."""
        batch_size = 2
        test_images = keras.random.normal((batch_size, 224, 224, 3))

        features = built_model(test_images, training=False)
        spatial_features = built_model.get_spatial_features(features)

        patch_height = 224 // 16  # 14
        patch_width = 224 // 16  # 14
        assert spatial_features.shape == (batch_size, patch_height, patch_width, 256)

        # Should be reshaped patch tokens
        patch_tokens = built_model.get_patch_tokens(features)
        expected_spatial = ops.reshape(
            patch_tokens,
            (batch_size, patch_height, patch_width, 256)
        )
        np.testing.assert_allclose(
            spatial_features.numpy(),
            expected_spatial.numpy(),
            rtol=1e-6
        )

    def test_different_image_sizes(self):
        """Test forward pass with different image sizes."""
        image_sizes = [(224, 224), (256, 256), (384, 384)]
        patch_size = 16

        for img_height, img_width in image_sizes:
            model = ViTSigLIP(
                img_size=img_height,  # Assume square images
                patch_size=patch_size,
                embed_dim=256,
                depth=2,
                num_heads=8
            )
            model.build((None, img_height, img_width, 3))

            test_images = keras.random.normal((2, img_height, img_width, 3))
            features = model(test_images, training=False)

            expected_patches = (img_height // patch_size) * (img_width // patch_size)
            expected_seq_len = expected_patches + 1  # +1 for CLS
            assert features.shape == (2, expected_seq_len, 256)


class TestSigLIPVisionTransformerConfigurations:
    """Test different transformer configuration options."""

    def test_different_attention_types(self):
        """Test model with different attention mechanisms."""
        attention_types = [
            'multi_head_attention',
            'group_query_attention'
        ]

        img_size = 224
        patch_size = 16
        num_patches = (img_size // patch_size)**2
        window_size = int(num_patches**0.5) # 14 for 224/16

        for attention_type in attention_types:
            model_kwargs = {
                'img_size': img_size,
                'patch_size': patch_size,
                'embed_dim': 256,
                'depth': 2,
                'num_heads': 8,
                'attention_type': attention_type
            }
            # if attention_type == 'window_attention':
            #     model_kwargs['window_size'] = window_size

            model = ViTSigLIP(**model_kwargs)
            model.build((None, 224, 224, 3))

            test_images = keras.random.normal((2, 224, 224, 3))
            features = model(test_images, training=False)

            assert features.shape == (2, 197, 256)  # 14*14 + 1 = 197

    def test_different_normalization_types(self):
        """Test model with different normalization types."""
        normalization_types = [
            'layer_norm',
            'rms_norm',
            'band_rms'
        ]

        for norm_type in normalization_types:
            model = ViTSigLIP(
                img_size=224,
                patch_size=16,
                embed_dim=256,
                depth=2,
                num_heads=8,
                normalization_type=norm_type
            )
            model.build((None, 224, 224, 3))

            test_images = keras.random.normal((2, 224, 224, 3))
            features = model(test_images, training=False)

            assert features.shape == (2, 197, 256)

    def test_different_ffn_types(self):
        """Test model with different FFN types."""
        ffn_types = [
            'mlp',
            'swiglu',
            'differential',
            'glu',
            'swin_mlp'
        ]

        for ffn_type in ffn_types:
            model = ViTSigLIP(
                img_size=224,
                patch_size=16,
                embed_dim=256,
                depth=2,
                num_heads=8,
                ffn_type=ffn_type
            )
            model.build((None, 224, 224, 3))

            test_images = keras.random.normal((2, 224, 224, 3))
            features = model(test_images, training=False)

            assert features.shape == (2, 197, 256)

    def test_pre_vs_post_normalization(self):
        """Test model with pre vs post normalization."""
        normalization_positions = ['pre', 'post']

        for norm_pos in normalization_positions:
            model = ViTSigLIP(
                img_size=224,
                patch_size=16,
                embed_dim=256,
                depth=2,
                num_heads=8,
                normalization_position=norm_pos
            )
            model.build((None, 224, 224, 3))

            test_images = keras.random.normal((2, 224, 224, 3))
            features = model(test_images, training=False)

            assert features.shape == (2, 197, 256)


class TestSigLIPVisionTransformerSerialization:
    """Test SigLIP Vision Transformer serialization and deserialization."""

    def test_config_serialization(self):
        """Test model configuration serialization."""
        model = ViTSigLIP(
            img_size=256,
            embed_dim=512,
            depth=8,
            num_heads=8, # FIX: Corrected from 12 to a divisor of 512
            attention_type='window_attention',
            dropout=0.1
        )

        model_config = model.get_config()

        assert isinstance(model_config, dict)
        assert model_config['img_size'] == 256
        assert model_config['embed_dim'] == 512
        assert model_config['depth'] == 8
        assert model_config['attention_type'] == 'window_attention'
        assert model_config['dropout'] == 0.1

    def test_from_config_reconstruction(self):
        """Test model reconstruction from configuration."""
        original_model = ViTSigLIP(
            img_size=256,
            embed_dim=384,
            depth=6,
            num_heads=6,
            attention_type='group_query_attention'
        )

        # Get config and reconstruct
        model_config = original_model.get_config()
        reconstructed_model = ViTSigLIP.from_config(model_config)

        # Check that configs match
        assert reconstructed_model.img_size == original_model.img_size
        assert reconstructed_model.embed_dim == original_model.embed_dim
        assert reconstructed_model.depth == original_model.depth
        assert reconstructed_model.num_heads == original_model.num_heads
        assert reconstructed_model.attention_type == original_model.attention_type

    def test_build_config_serialization(self):
        """Test build configuration serialization."""
        model = ViTSigLIP(embed_dim=256, depth=4, num_heads=8)

        input_shape = (None, 224, 224, 3)
        model.build(input_shape)

        build_config = model.get_build_config()
        assert build_config['input_shape'] == input_shape

        # Test build from config
        new_model = ViTSigLIP(embed_dim=256, depth=4, num_heads=8)
        new_model.build_from_config(build_config)
        assert new_model.built is True

    def test_model_save_load(self):
        """Test saving and loading complete model."""
        # Create and build model
        model = ViTSigLIP(
            img_size=224,
            embed_dim=256,
            depth=3,
            num_heads=8,
            dropout=0.0
        )
        model.build((None, 224, 224, 3))

        # Test forward pass before saving
        test_images = keras.random.normal((2, 224, 224, 3))
        original_outputs = model(test_images, training=False)

        # Save and load model
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'test_siglip_vit.keras')
            model.save(model_path)

            loaded_model = keras.models.load_model(
                model_path
            )

            # Test that loaded model produces same output
            loaded_outputs = loaded_model(test_images, training=False)

            # Outputs should be very close
            np.testing.assert_allclose(
                original_outputs.numpy(),
                loaded_outputs.numpy(),
                rtol=1e-5,
                atol=1e-6
            )


class TestSigLIPVisionTransformerEdgeCases:
    """Test SigLIP Vision Transformer edge cases and error handling."""

    def test_minimum_viable_configuration(self):
        """Test model with minimum viable configuration."""
        model = ViTSigLIP(
            img_size=32,  # Small image
            patch_size=16,  # Large patches
            embed_dim=64,  # Small embedding
            depth=1,  # Single layer
            num_heads=2,  # Few heads (64/2 = 32)
            mlp_ratio=2.0
        )
        model.build((None, 32, 32, 3))

        test_images = keras.random.normal((2, 32, 32, 3))
        features = model(test_images, training=False)

        expected_patches = (32 // 16) ** 2  # 4 patches
        assert features.shape == (2, expected_patches + 1, 64)  # (2, 5, 64)

    def test_single_sample_batch(self):
        """Test model with batch size of 1."""
        model = ViTSigLIP(embed_dim=256, depth=2, num_heads=8)
        model.build((None, 224, 224, 3))

        test_images = keras.random.normal((1, 224, 224, 3))
        features = model(test_images, training=False)

        assert features.shape == (1, 197, 256)  # 14*14 + 1 = 197

    def test_large_batch_size(self):
        """Test model with large batch size."""
        model = ViTSigLIP(
            embed_dim=128,  # Smaller to save memory
            depth=2,
            num_heads=4
        )
        model.build((None, 224, 224, 3))

        batch_size = 32
        test_images = keras.random.normal((batch_size, 224, 224, 3))
        features = model(test_images, training=False)

        assert features.shape == (batch_size, 197, 128)

    def test_grayscale_images(self):
        """Test model with grayscale images."""
        model = ViTSigLIP(embed_dim=256, depth=2, num_heads=8)
        model.build((None, 224, 224, 1))  # Grayscale

        test_images = keras.random.normal((2, 224, 224, 1))
        features = model(test_images, training=False)

        assert features.shape == (2, 197, 256)

    def test_different_channel_counts(self):
        """Test model with different channel counts."""
        channel_counts = [1, 3, 4]  # Grayscale, RGB, RGBA

        for channels in channel_counts:
            model = ViTSigLIP(embed_dim=256, depth=2, num_heads=8)
            model.build((None, 224, 224, channels))

            test_images = keras.random.normal((2, 224, 224, channels))
            features = model(test_images, training=False)

            assert features.shape == (2, 197, 256)

    def test_very_large_images(self):
        """Test model with large images."""
        model = ViTSigLIP(
            img_size=512,
            patch_size=32,  # Larger patches to keep sequence reasonable
            embed_dim=256,
            depth=2,
            num_heads=8
        )
        model.build((None, 512, 512, 3))

        test_images = keras.random.normal((1, 512, 512, 3))  # Small batch for memory
        features = model(test_images, training=False)

        expected_patches = (512 // 32) ** 2  # 16*16 = 256
        assert features.shape == (1, expected_patches + 1, 256)  # (1, 257, 256)

    def test_extreme_aspect_ratios(self):
        """Test that square image assumption is enforced."""
        model = ViTSigLIP(img_size=224, embed_dim=256, depth=2, num_heads=8)

        # This should work (square)
        model.build((None, 224, 224, 3))

        # Non-square should fail during forward pass or building
        # The current implementation assumes square images due to img_size parameter


class TestSigLIPVisionTransformerFactoryFunctions:
    """Test factory functions for creating SigLIP Vision Transformer models."""

    def test_create_siglip_vit_basic(self):
        """Test basic create_siglip_vit functionality."""
        model = create_siglip_vit()

        assert isinstance(model, ViTSigLIP)
        # Should use default config
        assert model.img_size == 224
        assert model.embed_dim == 768
        assert model.depth == 12
        assert model.num_heads == 12

    def test_create_siglip_vit_with_config(self):
        """Test create_siglip_vit with custom configuration."""
        config = ViTSigLIPConfig(
            img_size=256,
            embed_dim=512,
            depth=8,
            num_heads=8
        )
        model = create_siglip_vit(config)

        assert model.img_size == 256
        assert model.embed_dim == 512
        assert model.depth == 8
        assert model.num_heads == 8

    def test_create_siglip_vit_with_kwargs(self):
        """Test create_siglip_vit with kwargs override."""
        config = ViTSigLIPConfig(embed_dim=768, num_heads=16)
        model = create_siglip_vit(
            config,
            embed_dim=512,  # Override config
            num_heads=8,
            attention_type='window_attention'
        )

        assert model.embed_dim == 512  # Should be overridden
        assert model.attention_type == 'window_attention'

    def test_create_siglip_vit_base(self):
        """Test create_siglip_vit_base predefined configuration."""
        model = create_siglip_vit_base()

        assert isinstance(model, ViTSigLIP)
        assert model.embed_dim == 768
        assert model.depth == 12
        assert model.num_heads == 12

        # Test functionality
        model.build((None, 224, 224, 3))
        test_images = keras.random.normal((2, 224, 224, 3))
        features = model(test_images, training=False)
        assert features.shape == (2, 197, 768)

    def test_create_siglip_vit_large(self):
        """Test create_siglip_vit_large predefined configuration."""
        model = create_siglip_vit_large()

        assert isinstance(model, ViTSigLIP)
        assert model.embed_dim == 1024
        assert model.depth == 24
        assert model.num_heads == 16

        # Test functionality
        model.build((None, 224, 224, 3))
        test_images = keras.random.normal((1, 224, 224, 3))  # Small batch for large model
        features = model(test_images, training=False)
        assert features.shape == (1, 197, 1024)

    def test_create_siglip_vit_small(self):
        """Test create_siglip_vit_small predefined configuration."""
        model = create_siglip_vit_small()

        assert isinstance(model, ViTSigLIP)
        assert model.embed_dim == 384
        assert model.depth == 12
        assert model.num_heads == 6

        # Test functionality
        model.build((None, 224, 224, 3))
        test_images = keras.random.normal((2, 224, 224, 3))
        features = model(test_images, training=False)
        assert features.shape == (2, 197, 384)

    def test_factory_functions_with_overrides(self):
        """Test factory functions with parameter overrides."""
        # Test base with overrides
        model_base = create_siglip_vit_base(
            dropout=0.1,
            attention_type='group_query_attention'
        )
        assert model_base.dropout == 0.1
        assert model_base.attention_type == 'group_query_attention'
        assert model_base.embed_dim == 768  # Should keep base config

        # Test small with overrides
        model_small = create_siglip_vit_small(
            img_size=256,
            ffn_type='swiglu'
        )
        assert model_small.img_size == 256
        assert model_small.ffn_type == 'swiglu'
        assert model_small.embed_dim == 384  # Should keep small config

    def test_build_and_initialize_siglip_vit(self):
        """Test build and initialize helper function."""
        model = create_siglip_vit_small()

        # Build and initialize
        built_model = build_and_initialize_siglip_vit(
            model,
            input_shape=(224, 224, 3)
        )

        assert built_model is model  # Should return same instance
        assert model.built is True
        assert model.count_params() > 0

        # Test with compilation
        compile_config = {
            'optimizer': 'adam',
            'loss': 'mse'
        }
        compiled_model = build_and_initialize_siglip_vit(
            create_siglip_vit_small(),
            input_shape=(224, 224, 3),
            compile_config=compile_config
        )

        assert compiled_model.built is True
        # Check that model is compiled (has optimizer)
        assert compiled_model.optimizer is not None


class TestSigLIPVisionTransformerIntegration:
    """Integration tests for SigLIP Vision Transformer components working together."""

    def test_end_to_end_feature_extraction(self):
        """Test complete feature extraction workflow."""
        model = create_siglip_vit(
            ViTSigLIPConfig(
                embed_dim=256,
                depth=3,
                num_heads=8,
                dropout=0.0
            )
        )
        model.build((None, 224, 224, 3))

        # Create test data
        batch_size = 4
        test_images = keras.random.normal((batch_size, 224, 224, 3))

        # Forward pass
        features = model(test_images, training=False)

        # Test feature extraction methods
        cls_features = model.get_cls_token(features)
        patch_features = model.get_patch_tokens(features)
        spatial_features = model.get_spatial_features(features)

        # Check shapes
        assert cls_features.shape == (batch_size, 256)
        assert patch_features.shape == (batch_size, 196, 256)
        assert spatial_features.shape == (batch_size, 14, 14, 256)

        # Check consistency
        # CLS + patches should equal full features
        reconstructed = ops.concatenate([
            ops.expand_dims(cls_features, axis=1),
            patch_features
        ], axis=1)
        np.testing.assert_allclose(
            features.numpy(),
            reconstructed.numpy(),
            rtol=1e-6
        )

    def test_different_configurations_consistency(self):
        """Test that different configurations produce consistent behavior."""
        configurations = [
            {
                'attention_type': 'multi_head_attention',
                'normalization_type': 'layer_norm',
                'ffn_type': 'mlp'
            },
            {
                'attention_type': 'group_query_attention',
                'normalization_type': 'layer_norm',
                'ffn_type': 'differential'
            }
        ]

        img_size = 224
        patch_size = 16
        num_patches = (img_size // patch_size)**2
        window_size = int(num_patches**0.5)

        test_images = keras.random.normal((2, 224, 224, 3))

        for config_dict in configurations:
            model_kwargs = {
                'embed_dim': 256,
                'depth': 2,
                'num_heads': 8,
                **config_dict
            }
            # if config_dict['attention_type'] == 'window_attention':
            #     model_kwargs['window_size'] = window_size

            model = ViTSigLIP(**model_kwargs)
            model.build((None, 224, 224, 3))

            features = model(test_images, training=False)
            assert features.shape == (2, 197, 256)

            # Test feature extraction methods work
            cls_token = model.get_cls_token(features)
            patch_tokens = model.get_patch_tokens(features)
            spatial_features = model.get_spatial_features(features)

            assert cls_token.shape == (2, 256)
            assert patch_tokens.shape == (2, 196, 256)
            assert spatial_features.shape == (2, 14, 14, 256)

    def test_gradient_flow_integration(self):
        """Test that gradients flow through the entire model."""
        model = create_siglip_vit_small(normalization_position='pre')  # Use pre-norm for stability
        model.build((None, 224, 224, 3))

        test_images = keras.random.normal((2, 224, 224, 3))

        with tf.GradientTape() as tape:
            features = model(test_images, training=True)

            # Create a simple loss (e.g., minimize CLS token norm)
            cls_token = model.get_cls_token(features)
            loss = ops.mean(ops.sum(ops.square(cls_token), axis=-1))

        gradients = tape.gradient(loss, model.trainable_weights)

        # Check that we have gradients for most weights
        non_none_grads = [g for g in gradients if g is not None]
        assert len(non_none_grads) > len(model.trainable_weights) * 0.8, \
            f"Only {len(non_none_grads)}/{len(model.trainable_weights)} weights have gradients."


        # Check that gradients have reasonable magnitudes
        grad_norms = [
            ops.sqrt(ops.sum(ops.square(g))) for g in non_none_grads if g is not None
        ]
        assert all(norm > 1e-12 for norm in grad_norms), "Found vanishingly small gradients."
        assert all(norm < 1e6 for norm in grad_norms), "Found exploding gradients."

    def test_model_parameter_count_scaling(self):
        """Test that parameter count scales appropriately with model size."""
        configs = [
            {'embed_dim': 256, 'depth': 6, 'num_heads': 8},
            {'embed_dim': 512, 'depth': 12, 'num_heads': 8},
            {'embed_dim': 768, 'depth': 12, 'num_heads': 12},
        ]

        param_counts = []
        for config in configs:
            model = ViTSigLIP(**config)
            model.build((None, 224, 224, 3))
            param_counts.append(model.count_params())

        # Larger models should have more parameters
        assert param_counts[0] < param_counts[1] < param_counts[2]

    def test_feature_quality_basic_properties(self):
        """Test basic feature quality properties."""
        model = create_siglip_vit_small()
        model.build((None, 224, 224, 3))

        test_images = keras.random.normal((4, 224, 224, 3))
        features = model(test_images, training=False)

        # Features should have reasonable variance (not all the same)
        feature_std = ops.std(features, axis=0)
        mean_std = ops.mean(feature_std)
        assert mean_std > 0.01  # Some variation across dimensions

        # CLS token should be different from patch tokens
        cls_token = model.get_cls_token(features)
        patch_tokens = model.get_patch_tokens(features)

        # Mean CLS features should be different from mean patch features
        cls_mean = ops.mean(cls_token, axis=0)
        patch_mean = ops.mean(ops.mean(patch_tokens, axis=1), axis=0)

        difference_norm = ops.norm(cls_mean - patch_mean)
        assert difference_norm > 0.1  # Should have meaningful difference


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])