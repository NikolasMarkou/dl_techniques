import pytest
import numpy as np
import tensorflow as tf
import keras
from keras import ops, layers, models
import tempfile
import os
from typing import Any, Dict

from dl_techniques.layers.transformers.vision_encoder import (
    VisionEncoder,
    create_vision_encoder,
    create_vit_encoder,
    create_siglip_encoder
)


# --- Test Class ---
class TestVisionEncoder:
    """
    Comprehensive and modern test suite for the VisionEncoder.
    This suite follows modern Keras 3 testing best practices and covers all
    architectural variations and factory patterns.
    """

    # --- Fixtures for Reusability ---
    @pytest.fixture
    def basic_config(self) -> Dict[str, Any]:
        """Provides a basic configuration for a small, testable encoder."""
        return {
            'img_size': 32,
            'patch_size': 8,
            'embed_dim': 64,
            'depth': 2,
            'num_heads': 4,
        }

    @pytest.fixture
    def vit_config(self) -> Dict[str, Any]:
        """Provides ViT-style configuration."""
        return {
            'img_size': 224,
            'patch_size': 16,
            'embed_dim': 768,
            'depth': 12,
            'num_heads': 12,
            'use_cls_token': True,
            'output_mode': 'cls'
        }

    @pytest.fixture
    def modern_config(self) -> Dict[str, Any]:
        """Provides modern encoder configuration with advanced features."""
        # Ensure num_patches matches window_size^2 for WindowAttention compatibility
        window_size = 4
        num_patches_per_dim = window_size
        patch_size = 8
        img_size = num_patches_per_dim * patch_size  # 4 * 8 = 32

        return {
            'img_size': img_size,
            'patch_size': patch_size,
            'embed_dim': 128,
            'depth': 4,
            'num_heads': 4,
            'patch_embed_type': 'siglip',
            'attention_type': 'window',
            'normalization_type': 'rms_norm',
            'normalization_position': 'pre',
            'ffn_type': 'swiglu',
            'stochastic_depth_rate': 0.1,
            'output_mode': 'mean',
            'use_cls_token': False,
            'attention_args': {'window_size': window_size}
        }

    @pytest.fixture
    def sample_images(self) -> tf.Tensor:
        """Provides a batch of sample images for testing."""
        return tf.random.uniform(
            shape=(2, 32, 32, 3), minval=0.0, maxval=1.0, dtype=tf.float32
        )

    # ===============================================
    # 1. Initialization and Build Tests
    # ===============================================
    def test_initialization_defaults(self, basic_config):
        """Tests encoder initialization with default parameters."""
        encoder = VisionEncoder(**basic_config)
        assert not encoder.built
        assert encoder.patch_embed_type == 'linear'
        assert encoder.attention_type == 'multi_head'
        assert encoder.normalization_type == 'layer_norm'
        assert encoder.ffn_type == 'mlp'
        assert encoder.output_mode == 'cls'
        assert encoder.use_cls_token

    @pytest.mark.parametrize("patch_embed_type", ['linear', 'siglip', 'conv', 'hybrid'])
    def test_initialization_patch_embed_types(self, basic_config, patch_embed_type):
        """Tests initialization with different patch embedding types."""
        config = {**basic_config, 'patch_embed_type': patch_embed_type}
        encoder = VisionEncoder(**config)
        assert encoder.patch_embed_type == patch_embed_type
        assert hasattr(encoder, 'patch_embed')

    @pytest.mark.parametrize("attention_type", [
        'multi_head', 'window', 'group_query', 'differential'
    ])
    def test_initialization_attention_types(self, basic_config, attention_type):
        """Tests initialization with different attention mechanisms."""
        config = {**basic_config, 'attention_type': attention_type}
        if attention_type == 'window':
            # This makes num_patches match window_size^2
            config['img_size'] = 16
            config['patch_size'] = 8
            config['attention_args'] = {'window_size': 2}
        encoder = VisionEncoder(**config)
        assert encoder.attention_type == attention_type

    def test_build_process(self, basic_config, sample_images):
        """Tests that encoder and all sub-layers are built correctly."""
        encoder = VisionEncoder(**basic_config)
        assert not encoder.built
        output = encoder(sample_images)
        assert encoder.built
        assert hasattr(encoder.patch_embed, 'built')
        assert encoder.patch_embed.built
        assert encoder.cls_token is not None
        assert encoder.cls_token.shape == (1, 1, basic_config['embed_dim'])

    def test_build_without_cls_token(self, basic_config, sample_images):
        """Tests that CLS token is not created when use_cls_token is False."""
        config = {**basic_config, 'use_cls_token': False, 'output_mode': 'mean'}
        encoder = VisionEncoder(**config)
        encoder(sample_images)
        assert encoder.cls_token is None

    # ===============================================
    # 2. Parameter Validation Tests
    # ===============================================
    def test_invalid_img_size_patch_size(self):
        """Tests validation of img_size and patch_size compatibility."""
        with pytest.raises(ValueError, match="img_size .* must be divisible by patch_size"):
            VisionEncoder(img_size=32, patch_size=7)

    def test_invalid_embed_dim_num_heads(self):
        """Tests validation of embed_dim and num_heads compatibility."""
        with pytest.raises(ValueError, match="embed_dim .* must be divisible by num_heads"):
            VisionEncoder(embed_dim=64, num_heads=5)

    def test_invalid_output_mode_cls_without_cls_token(self):
        """Tests validation of output_mode='cls' requiring use_cls_token=True."""
        with pytest.raises(ValueError, match="output_mode='cls' requires use_cls_token=True"):
            VisionEncoder(output_mode='cls', use_cls_token=False)

    # ===============================================
    # 3. Forward Pass and Core Behavior Tests
    # ===============================================
    @pytest.mark.parametrize("output_mode", ['cls', 'mean', 'max'])
    def test_forward_pass_pooled_output_modes(self, basic_config, sample_images, output_mode):
        """Tests forward pass with pooled output modes."""
        config = {**basic_config, 'output_mode': output_mode}
        encoder = VisionEncoder(**config)
        output = encoder(sample_images, training=False)
        expected_shape = (sample_images.shape[0], basic_config['embed_dim'])
        assert output.shape == expected_shape
        assert not np.any(np.isnan(ops.convert_to_numpy(output)))

    def test_forward_pass_sequence_output(self, basic_config, sample_images):
        """Tests forward pass with 'none' output mode."""
        config = {**basic_config, 'output_mode': 'none'}
        encoder = VisionEncoder(**config)
        output = encoder(sample_images, training=False)
        expected_shape = (sample_images.shape[0], encoder.seq_len, basic_config['embed_dim'])
        assert output.shape == expected_shape

    def test_forward_pass_no_cls_token(self, basic_config, sample_images):
        """Tests forward pass without CLS token."""
        config = {**basic_config, 'use_cls_token': False, 'output_mode': 'mean'}
        encoder = VisionEncoder(**config)
        output = encoder(sample_images, training=False)
        expected_shape = (sample_images.shape[0], basic_config['embed_dim'])
        assert output.shape == expected_shape
        assert encoder.seq_len == encoder.num_patches

    def test_training_vs_inference_modes(self, basic_config, sample_images):
        """Tests behavior difference between training and inference modes."""
        config = {**basic_config, 'dropout_rate': 0.5, 'pos_dropout_rate': 0.3}
        encoder = VisionEncoder(**config)
        output_train = encoder(sample_images, training=True)
        output_infer = encoder(sample_images, training=False)
        assert output_train.shape == output_infer.shape
        assert not np.allclose(ops.convert_to_numpy(output_train), ops.convert_to_numpy(output_infer))

    def test_get_cls_features(self, basic_config, sample_images):
        """Tests get_cls_features method."""
        encoder = VisionEncoder(**basic_config)
        cls_features = encoder.get_cls_features(sample_images, training=False)
        expected_shape = (sample_images.shape[0], basic_config['embed_dim'])
        assert cls_features.shape == expected_shape

    def test_get_patch_features(self, basic_config, sample_images):
        """Tests get_patch_features method."""
        encoder = VisionEncoder(**basic_config)
        patch_features = encoder.get_patch_features(sample_images, training=False)
        expected_shape = (sample_images.shape[0], encoder.num_patches, basic_config['embed_dim'])
        assert patch_features.shape == expected_shape

    def test_get_spatial_features(self, basic_config, sample_images):
        """Tests get_spatial_features method."""
        encoder = VisionEncoder(**basic_config)
        spatial_features = encoder.get_spatial_features(sample_images, training=False)
        patches_per_dim = basic_config['img_size'] // basic_config['patch_size']
        expected_shape = (sample_images.shape[0], patches_per_dim, patches_per_dim, basic_config['embed_dim'])
        assert spatial_features.shape == expected_shape

    # ===============================================
    # 4. Serialization Tests (The Gold Standard)
    # ===============================================
    def test_full_serialization_cycle_basic(self, basic_config, sample_images):
        """Tests full serialization cycle with basic configuration."""
        inputs = layers.Input(shape=sample_images.shape[1:])
        outputs = VisionEncoder(**basic_config)(inputs)
        model = models.Model(inputs, outputs)
        original_prediction = model(sample_images, training=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_basic_vision_encoder.keras")
            model.save(filepath)
            loaded_model = models.load_model(filepath)
            loaded_prediction = loaded_model(sample_images, training=False)

            np.testing.assert_allclose(
                ops.convert_to_numpy(original_prediction),
                ops.convert_to_numpy(loaded_prediction),
                rtol=1e-6, atol=1e-6
            )

    def test_full_serialization_cycle_modern(self, modern_config):
        """Tests full serialization cycle with modern configuration."""
        img_size = modern_config['img_size']
        sample_images_modern = tf.random.uniform((2, img_size, img_size, 3))

        inputs = layers.Input(shape=sample_images_modern.shape[1:])
        outputs = VisionEncoder(**modern_config)(inputs)
        model = models.Model(inputs, outputs)
        original_prediction = model(sample_images_modern, training=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_modern_vision_encoder.keras")
            model.save(filepath)
            loaded_model = models.load_model(filepath)
            loaded_prediction = loaded_model(sample_images_modern, training=False)

            np.testing.assert_allclose(
                ops.convert_to_numpy(original_prediction),
                ops.convert_to_numpy(loaded_prediction),
                rtol=1e-6, atol=1e-6
            )

    # ===============================================
    # 5. Gradient and Training Integration Tests
    # ===============================================
    def test_gradient_flow(self, basic_config, sample_images):
        """Tests gradient flow through encoder."""
        encoder = VisionEncoder(**basic_config)
        x_var = tf.Variable(sample_images)

        with tf.GradientTape() as tape:
            output = encoder(x_var, training=True)
            loss = ops.mean(ops.square(output))

        gradients = tape.gradient(loss, encoder.trainable_variables)
        assert len(gradients) > 0, "No gradients were computed."
        assert all(g is not None for g in gradients), "A gradient is None."

    def test_model_training_loop_integration(self, basic_config):
        """Tests encoder integration in a standard training loop."""
        img_size = basic_config['img_size']
        model = models.Sequential([
            layers.InputLayer(shape=(img_size, img_size, 3)),
            VisionEncoder(**basic_config),
            layers.Dense(10)
        ])
        model.compile("adam", "sparse_categorical_crossentropy")
        x_train = tf.random.uniform((8, img_size, img_size, 3))
        y_train = tf.random.uniform((8,), maxval=10, dtype=tf.int32)
        history = model.fit(x_train, y_train, epochs=1, batch_size=4, verbose=0)
        assert 'loss' in history.history
        assert not np.isnan(history.history['loss'][0])

    # ===============================================
    # 6. Factory Functions Tests
    # ===============================================
    def test_create_vision_encoder_factory(self):
        """Tests the create_vision_encoder factory function."""
        encoder = create_vision_encoder(
            img_size=32, patch_size=8, embed_dim=64, depth=2, num_heads=4
        )
        assert isinstance(encoder, VisionEncoder)
        assert encoder.img_size == 32
        assert encoder.embed_dim == 64

    def test_create_vit_encoder_factory(self):
        """Tests the create_vit_encoder factory function."""
        encoder = create_vit_encoder(img_size=32, patch_size=8, embed_dim=64, depth=2, num_heads=4)
        assert isinstance(encoder, VisionEncoder)
        assert encoder.patch_embed_type == 'linear'
        assert encoder.attention_type == 'multi_head'
        assert encoder.use_cls_token
        assert encoder.output_mode == 'cls'

    def test_create_siglip_encoder_factory(self):
        """Tests the create_siglip_encoder factory function."""
        encoder = create_siglip_encoder(img_size=32, patch_size=8, embed_dim=64, depth=2, num_heads=4)
        assert isinstance(encoder, VisionEncoder)
        assert encoder.patch_embed_type == 'siglip'

    def test_factory_parameter_validation(self):
        """Tests that factory functions validate parameters properly."""
        with pytest.raises(ValueError, match="img_size .* must be divisible by patch_size"):
            create_vision_encoder(img_size=32, patch_size=7)

    # ===============================================
    # 7. Configuration and Get Config Tests
    # ===============================================
    def test_get_config_completeness(self, basic_config):
        """Tests that get_config contains all initialization parameters."""
        encoder = VisionEncoder(**basic_config)
        config = encoder.get_config()
        for key in basic_config:
            assert key in config, f"Missing {key} in get_config()"
        assert 'patch_embed_type' in config

    def test_config_reconstruction(self, basic_config):
        """Tests that an encoder can be reconstructed from its config."""
        original_encoder = VisionEncoder(**basic_config)
        config = original_encoder.get_config()
        reconstructed_encoder = VisionEncoder.from_config(config)
        assert reconstructed_encoder.img_size == original_encoder.img_size
        assert reconstructed_encoder.embed_dim == original_encoder.embed_dim
        assert reconstructed_encoder.depth == original_encoder.depth

    def test_compute_output_shape(self, basic_config):
        """Tests the compute_output_shape method."""
        encoder = VisionEncoder(**basic_config)
        input_shape = (None, 32, 32, 3)
        output_shape = encoder.compute_output_shape(input_shape)
        expected_shape = (None, basic_config['embed_dim'])  # Default is 'cls'
        assert output_shape == expected_shape

    def test_compute_output_shape_sequence(self, basic_config):
        """Tests compute_output_shape with 'none' output mode."""
        config = {**basic_config, 'output_mode': 'none'}
        encoder = VisionEncoder(**config)
        input_shape = (None, 32, 32, 3)
        output_shape = encoder.compute_output_shape(input_shape)
        expected_shape = (None, encoder.seq_len, basic_config['embed_dim'])
        assert output_shape == expected_shape

    # ===============================================
    # 8. Mixed Precision Compatibility Test
    # ===============================================
    def test_mixed_precision_compatibility(self, basic_config, sample_images):
        """Tests encoder compatibility with mixed precision training."""
        policy = keras.mixed_precision.Policy('mixed_float16')
        keras.mixed_precision.set_global_policy(policy)
        try:
            encoder = VisionEncoder(**basic_config)
            output = encoder(sample_images, training=False)
            assert output.dtype == tf.float16
        finally:
            keras.mixed_precision.set_global_policy('float32')