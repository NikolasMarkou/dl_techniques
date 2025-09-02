import pytest
import tempfile
import os
import numpy as np
import keras
import tensorflow as tf
from typing import Any, Dict

from dl_techniques.models.mobile_clip.model import MobileClipModel

class TestMobileClipModel:
    """Comprehensive test suite for the MobileClipModel."""

    @pytest.fixture
    def model_config(self) -> Dict[str, Any]:
        """Standard configuration for testing, based on a minimal S0 variant."""
        return {
            "embed_dim": 256,
            "image_config": {
                "backbone_name": "MobileNetV2",
                "backbone_weights": None,
                "backbone_trainable": True,
                "projection_dropout": 0.0,
            },
            "text_config": {
                "vocab_size": 1000,
                "max_seq_len": 32,
                "embed_dim": 128,
                "num_layers": 2,
                "num_heads": 4,
                "intermediate_size": 256,
                "dropout_rate": 0.0,
                "attention_dropout_rate": 0.0,
                "use_causal_mask": True,
            }
        }

    @pytest.fixture
    def sample_input(self) -> Dict[str, keras.KerasTensor]:
        """Sample input dictionary for testing."""
        return {
            'image': keras.random.normal(shape=(4, 224, 224, 3)),
            'text': keras.random.randint(shape=(4, 32), minval=0, maxval=1000, dtype="int32"),
        }

    def test_initialization(self, model_config):
        """Test model initialization and attribute setup."""
        model = MobileClipModel(**model_config)

        assert model.embed_dim == model_config['embed_dim']
        assert hasattr(model, 'image_encoder')
        assert hasattr(model, 'text_encoder')
        assert hasattr(model, 'logit_scale')
        assert model.image_encoder is not None
        assert model.text_encoder is not None
        assert model.image_encoder.projection_dim == model_config['embed_dim']
        assert model.text_encoder.projection_dim == model_config['embed_dim']

    def test_forward_pass(self, model_config, sample_input):
        """Test the forward pass with both image and text inputs."""
        model = MobileClipModel(**model_config)
        outputs = model(sample_input)

        assert isinstance(outputs, dict)
        assert 'image_features' in outputs
        assert 'text_features' in outputs
        assert 'logit_scale' in outputs

        image_features = outputs['image_features']
        text_features = outputs['text_features']

        assert image_features.shape == (sample_input['image'].shape[0], model_config['embed_dim'])
        assert text_features.shape == (sample_input['text'].shape[0], model_config['embed_dim'])

    def test_serialization_cycle(self, model_config, sample_input):
        """CRITICAL TEST: Full save/load serialization cycle."""
        # 1. Create original model
        model = MobileClipModel(**model_config)

        # 2. Get prediction from original
        original_outputs = model(sample_input)
        original_img_feat = original_outputs['image_features']
        original_txt_feat = original_outputs['text_features']

        # 3. Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_model.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_outputs = loaded_model(sample_input)
            loaded_img_feat = loaded_outputs['image_features']
            loaded_txt_feat = loaded_outputs['text_features']

            # 4. Verify identical outputs
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_img_feat),
                keras.ops.convert_to_numpy(loaded_img_feat),
                rtol=1e-6, atol=1e-6,
                err_msg="Image features differ after serialization"
            )
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_txt_feat),
                keras.ops.convert_to_numpy(loaded_txt_feat),
                rtol=1e-6, atol=1e-6,
                err_msg="Text features differ after serialization"
            )

    def test_config_completeness(self, model_config):
        """Test that get_config contains all __init__ parameters."""
        model = MobileClipModel(**model_config)
        config = model.get_config()

        for key in model_config:
            assert key in config, f"Missing '{key}' in get_config()"
        assert 'logit_scale_init' in config
        assert 'output_dict' in config

    def test_gradients_flow(self, model_config, sample_input):
        """Test that gradients flow through both encoders."""
        model = MobileClipModel(**model_config)

        with tf.GradientTape() as tape:
            outputs = model(sample_input, training=True)
            image_features = outputs['image_features']
            text_features = outputs['text_features']
            logit_scale = outputs['logit_scale']

            # Simple contrastive loss for testing
            logits = keras.ops.matmul(image_features, keras.ops.transpose(text_features)) * logit_scale
            labels = keras.ops.arange(sample_input['image'].shape[0])
            loss = keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
            loss = keras.ops.mean(loss)

        gradients = tape.gradient(loss, model.trainable_variables)

        assert len(gradients) == len(model.trainable_variables)
        assert all(g is not None for g in gradients)

    @pytest.mark.parametrize("training", [True, False])
    def test_training_modes(self, model_config, sample_input, training):
        """Test behavior in different training modes (e.g., for dropout)."""
        model = MobileClipModel(**model_config)
        outputs = model(sample_input, training=training)
        assert outputs['image_features'] is not None
        assert outputs['text_features'] is not None

    def test_edge_cases(self):
        """Test initialization with invalid arguments."""
        with pytest.raises(ValueError, match="embed_dim must be positive"):
            MobileClipModel.from_variant("s0", embed_dim=0)

        with pytest.raises(ValueError, match="Unknown variant"):
            MobileClipModel.from_variant("non_existent_variant")

        with pytest.raises(TypeError, match="image_config must be a dictionary"):
            MobileClipModel(embed_dim=512, image_config=None, text_config={})

    @pytest.mark.parametrize("variant", MobileClipModel.MODEL_VARIANTS.keys())
    def test_from_variant_factory(self, variant):
        """Test model creation from all predefined variants."""
        # Note: some backbones might not exist in a minimal test env
        # We will catch the specific error for that.
        try:
            model = MobileClipModel.from_variant(variant)
            assert isinstance(model, MobileClipModel)
            assert model.embed_dim == MobileClipModel.MODEL_VARIANTS[variant]['embed_dim']
        except ValueError as e:
            # Allow failure only if it's because the backbone isn't supported
            # in the test environment (e.g. custom ViT or MCI models).
            if "Unsupported backbone" in str(e):
                pytest.skip(f"Backbone for variant '{variant}' not available in test environment.")
            else:
                raise e

    def test_output_format(self, model_config, sample_input):
        """Test both dictionary and tuple output formats."""
        # Test dictionary output (default)
        model_dict = MobileClipModel(**model_config, output_dict=True)
        output_dict = model_dict(sample_input)
        assert isinstance(output_dict, dict)

        # Test tuple output
        model_tuple = MobileClipModel(**model_config, output_dict=False)
        output_tuple = model_tuple(sample_input)
        assert isinstance(output_tuple, tuple)
        assert len(output_tuple) == 3  # image, text, scale

    def test_partial_input(self, model_config, sample_input):
        """Test model with only image or only text input."""
        model = MobileClipModel(**model_config)

        # Image only
        image_only_input = {'image': sample_input['image']}
        image_only_output = model(image_only_input)
        assert image_only_output['image_features'] is not None
        assert image_only_output['text_features'] is None

        # Text only
        text_only_input = {'text': sample_input['text']}
        text_only_output = model(text_only_input)
        assert text_only_output['image_features'] is None
        assert text_only_output['text_features'] is not None

    def test_encode_methods(self, model_config, sample_input):
        """Test the standalone encode_image and encode_text methods."""
        model = MobileClipModel(**model_config)

        # Test image encoding
        image_features = model.encode_image(sample_input['image'])
        assert image_features.shape == (sample_input['image'].shape[0], model_config['embed_dim'])

        # Test text encoding
        text_features = model.encode_text(sample_input['text'])
        assert text_features.shape == (sample_input['text'].shape[0], model_config['embed_dim'])

if __name__ == "__main__":
    # To run this test file, save it as `test_mobile_clip_model.py`
    # and run pytest from your terminal in the project root directory.
    pytest.main([__file__, "-v", "--tb=short"])