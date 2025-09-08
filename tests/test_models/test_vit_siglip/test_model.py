"""
Comprehensive test suite for the SigLIP Vision Transformer implementation.

This test suite covers all aspects of the SigLIP ViT model following modern Keras 3 patterns,
including factory-based component creation, proper serialization, robust parameter
validation, two-stage patch embedding, and integration testing.
"""

import os
import keras
import pytest
import tempfile
import numpy as np
from typing import Tuple, Dict, Any, List

from dl_techniques.models.vit_siglip.model import (
    SigLIPVisionTransformer,
    create_siglip_vision_transformer
)


class TestSigLIPInitialization:
    """Test suite for SigLIP ViT model initialization with modern patterns."""

    def test_default_initialization(self):
        """Test initialization with default parameters."""
        model = SigLIPVisionTransformer()

        # Check default values
        assert model.input_shape_config == (224, 224, 3)
        assert model.num_classes == 1000
        assert model.scale == "base"
        assert model.patch_size == (16, 16)
        assert model.include_top is True
        assert model.pooling is None
        assert model.dropout_rate == 0.0
        assert model.attention_dropout_rate == 0.0
        assert model.pos_dropout_rate == 0.0
        assert model.normalization_type == "layer_norm"
        assert model.normalization_position == "post"
        assert model.ffn_type == "mlp"
        assert model.activation == "gelu"
        assert not model.built

    def test_custom_initialization(self):
        """Test initialization with custom parameters."""
        model = SigLIPVisionTransformer(
            input_shape=(384, 384, 3),
            num_classes=10,
            scale="small",
            patch_size=(32, 32),
            include_top=False,
            pooling="cls",
            dropout_rate=0.1,
            attention_dropout_rate=0.05,
            pos_dropout_rate=0.2,
            normalization_type="rms_norm",
            normalization_position="pre",
            ffn_type="swiglu",
            activation="relu",
            name="custom_siglip_vit"
        )

        # Check custom values
        assert model.input_shape_config == (384, 384, 3)
        assert model.num_classes == 10
        assert model.scale == "small"
        assert model.patch_size == (32, 32)
        assert model.include_top is False
        assert model.pooling == "cls"
        assert model.dropout_rate == 0.1
        assert model.attention_dropout_rate == 0.05
        assert model.pos_dropout_rate == 0.2
        assert model.normalization_type == "rms_norm"
        assert model.normalization_position == "pre"
        assert model.ffn_type == "swiglu"
        assert model.activation == "relu"
        assert model.name == "custom_siglip_vit"

    def test_scale_configurations(self):
        """Test all SigLIP scale configurations."""
        scales = ["tiny", "small", "base", "large", "huge"]
        expected_configs = {
            "tiny": (192, 3, 12, 4.0),
            "small": (384, 6, 12, 4.0),
            "base": (768, 12, 12, 4.0),
            "large": (1024, 16, 24, 4.0),
            "huge": (1280, 16, 32, 4.0),
        }

        for scale in scales:
            model = SigLIPVisionTransformer(scale=scale)
            embed_dim, num_heads, num_layers, mlp_ratio = expected_configs[scale]

            assert model.embed_dim == embed_dim
            assert model.num_heads == num_heads
            assert model.num_layers == num_layers
            assert model.mlp_ratio == mlp_ratio
            assert model.intermediate_size == int(embed_dim * mlp_ratio)

    def test_patch_size_handling(self):
        """Test different patch size formats."""
        # Integer patch size
        model1 = SigLIPVisionTransformer(patch_size=16)
        assert model1.patch_size == (16, 16)

        # Tuple patch size with compatible dimensions
        model2 = SigLIPVisionTransformer(input_shape=(96, 144, 3), patch_size=(8, 12))
        assert model2.patch_size == (8, 12)

    def test_invalid_parameters(self):
        """Test that invalid parameters raise appropriate errors."""
        # Invalid input_shape
        with pytest.raises(ValueError, match="input_shape must be a 3-tuple"):
            SigLIPVisionTransformer(input_shape=(224, 224))

        with pytest.raises(ValueError, match="All input_shape dimensions must be positive"):
            SigLIPVisionTransformer(input_shape=(224, -224, 3))

        # Invalid scale
        with pytest.raises(ValueError, match="Unsupported scale"):
            SigLIPVisionTransformer(scale="invalid_scale")

        # Invalid pooling
        with pytest.raises(ValueError, match="Unsupported pooling"):
            SigLIPVisionTransformer(pooling="invalid_pooling")

        # Invalid patch size
        with pytest.raises(ValueError, match="patch_size must be positive"):
            SigLIPVisionTransformer(patch_size=-16)

        with pytest.raises(ValueError, match="patch_size dimensions must be positive"):
            SigLIPVisionTransformer(patch_size=(16, -16))

        # Incompatible image and patch dimensions
        with pytest.raises(ValueError, match="Image height .* must be divisible by patch height"):
            SigLIPVisionTransformer(input_shape=(225, 224, 3), patch_size=16)

        # Invalid dropout rates
        with pytest.raises(ValueError, match="dropout_rate must be between 0 and 1"):
            SigLIPVisionTransformer(dropout_rate=1.5)

        with pytest.raises(ValueError, match="attention_dropout_rate must be between 0 and 1"):
            SigLIPVisionTransformer(attention_dropout_rate=-0.1)

        # Invalid num_classes
        with pytest.raises(ValueError, match="num_classes must be positive"):
            SigLIPVisionTransformer(num_classes=0)


class TestSigLIPBuildProcess:
    """Test suite for SigLIP ViT model building process with modern patterns."""

    @pytest.fixture
    def sample_input_shapes(self) -> List[Tuple[int, int, int]]:
        """Sample input shapes for testing."""
        return [
            (224, 224, 3),
            (384, 384, 3),
            (128, 128, 1),
            (512, 512, 3)
        ]

    def test_build_process(self, sample_input_shapes):
        """Test that the model builds properly using modern patterns."""
        for input_shape in sample_input_shapes:
            model = SigLIPVisionTransformer(input_shape=input_shape, scale="tiny")  # Use tiny for speed

            # Model should not be built initially
            assert not model.built

            # Build the model
            batch_input_shape = (None,) + input_shape
            model.build(batch_input_shape)

            # Check that model is built
            assert model.built
            assert len(model.weights) > 0

            # Check that all sub-layers exist and are created properly
            assert model.siglip_patch_embed is not None  # SigLIP-specific two-stage embedding
            assert model.cls_token is not None
            assert model.pos_embed is not None
            assert len(model.transformer_layers) == model.num_layers

    def test_siglip_patch_embedding_structure(self):
        """Test that SigLIP two-stage patch embedding is properly structured."""
        model = SigLIPVisionTransformer(input_shape=(224, 224, 3), scale="tiny")
        model.build((None, 224, 224, 3))

        # SigLIP should have a Sequential patch embedding with specific structure
        assert isinstance(model.siglip_patch_embed, keras.Sequential)

        # Should have the two-stage structure: Conv2D -> LayerNorm -> Activation -> Conv2D
        layers = model.siglip_patch_embed.layers
        assert len(layers) == 4

        # First conv layer (stage 1)
        assert isinstance(layers[0], keras.layers.Conv2D)
        assert layers[0].filters == model.embed_dim // 2

        # LayerNorm
        assert isinstance(layers[1], keras.layers.LayerNormalization)

        # GELU activation
        assert isinstance(layers[2], keras.layers.Activation)

        # Second conv layer (stage 2)
        assert isinstance(layers[3], keras.layers.Conv2D)
        assert layers[3].filters == model.embed_dim

    def test_build_with_include_top(self):
        """Test building with classification head."""
        model = SigLIPVisionTransformer(include_top=True, num_classes=10, scale="tiny")
        model.build((None, 224, 224, 3))

        assert model.head is not None
        assert model.head.units == 10

    def test_build_without_include_top(self):
        """Test building without classification head."""
        model = SigLIPVisionTransformer(include_top=False, scale="tiny")
        model.build((None, 224, 224, 3))

        assert model.head is None

    def test_build_with_different_pooling(self):
        """Test building with different pooling options."""
        pooling_options = ["cls", "mean", "max", None]

        for pooling in pooling_options:
            model = SigLIPVisionTransformer(include_top=False, pooling=pooling, scale="tiny")
            model.build((None, 224, 224, 3))

            if pooling in ["mean", "max"]:
                assert model.global_pool is not None
            else:
                assert model.global_pool is None

    def test_build_normalization_handling(self):
        """Test that normalization is only created for pre-norm configuration."""
        # Pre-norm should have final normalization layer
        model_pre = SigLIPVisionTransformer(normalization_position="pre", scale="tiny")
        model_pre.build((None, 224, 224, 3))
        assert model_pre.norm is not None

        # Post-norm should NOT have final normalization layer
        model_post = SigLIPVisionTransformer(normalization_position="post", scale="tiny")
        model_post.build((None, 224, 224, 3))
        assert model_post.norm is None

    def test_build_idempotency(self):
        """Test that multiple build calls are idempotent."""
        model = SigLIPVisionTransformer(scale="tiny")
        input_shape = (None, 224, 224, 3)

        # Build multiple times
        model.build(input_shape)
        weights_after_first_build = len(model.weights)

        model.build(input_shape)
        weights_after_second_build = len(model.weights)

        # Should be the same
        assert weights_after_first_build == weights_after_second_build


class TestSigLIPOutputShapes:
    """Test suite for SigLIP ViT output shape computation."""

    @pytest.fixture
    def input_shapes_and_configs(self) -> List[Dict[str, Any]]:
        """Test configurations with expected shapes."""
        return [
            {
                "input_shape": (224, 224, 3),
                "patch_size": 16,
                "scale": "tiny",
                "expected_patches": 196,  # (224/16)^2
                "expected_seq_len": 197  # 196 + 1 (CLS)
            },
            {
                "input_shape": (384, 384, 3),
                "patch_size": (32, 32),
                "scale": "small",
                "expected_patches": 144,  # (384/32)^2
                "expected_seq_len": 145  # 144 + 1 (CLS)
            },
            {
                "input_shape": (128, 128, 1),
                "patch_size": 8,
                "scale": "tiny",
                "expected_patches": 256,  # (128/8)^2
                "expected_seq_len": 257  # 256 + 1 (CLS)
            }
        ]

    def test_compute_output_shape_with_top(self, input_shapes_and_configs):
        """Test output shape computation with classification head."""
        for config in input_shapes_and_configs:
            model = SigLIPVisionTransformer(
                input_shape=config["input_shape"],
                patch_size=config["patch_size"],
                scale=config["scale"],
                num_classes=10,
                include_top=True
            )

            batch_input_shape = (None,) + config["input_shape"]
            output_shape = model.compute_output_shape(batch_input_shape)

            assert output_shape == (None, 10)

    def test_compute_output_shape_without_top(self, input_shapes_and_configs):
        """Test output shape computation without classification head."""
        for config in input_shapes_and_configs:
            embed_dim = SigLIPVisionTransformer.SCALE_CONFIGS[config["scale"]][0]

            # Test different pooling options
            pooling_tests = [
                ("cls", (None, embed_dim)),
                ("mean", (None, embed_dim)),
                ("max", (None, embed_dim)),
                (None, (None, config["expected_seq_len"], embed_dim))
            ]

            for pooling, expected_shape in pooling_tests:
                model = SigLIPVisionTransformer(
                    input_shape=config["input_shape"],
                    patch_size=config["patch_size"],
                    scale=config["scale"],
                    include_top=False,
                    pooling=pooling
                )

                batch_input_shape = (None,) + config["input_shape"]
                output_shape = model.compute_output_shape(batch_input_shape)

                assert output_shape == expected_shape

    def test_invalid_input_shape_for_compute_output_shape(self):
        """Test compute_output_shape with invalid input shapes."""
        model = SigLIPVisionTransformer(scale="tiny")

        # 3D input shape should fail
        with pytest.raises(ValueError, match="Expected 4D input shape"):
            model.compute_output_shape((None, 224, 224))


class TestSigLIPForwardPass:
    """Test suite for SigLIP ViT forward pass with two-stage patch embedding."""

    @pytest.fixture
    def sample_inputs(self) -> Dict[str, np.ndarray]:
        """Sample inputs for testing."""
        return {
            "small": np.random.rand(2, 224, 224, 3).astype('float32'),
            "medium": np.random.rand(4, 384, 384, 3).astype('float32'),
            "grayscale": np.random.rand(2, 128, 128, 1).astype('float32')
        }

    def test_forward_pass_with_top(self, sample_inputs):
        """Test forward pass with classification head."""
        for input_name, input_tensor in sample_inputs.items():
            batch_size = input_tensor.shape[0]
            input_shape = input_tensor.shape[1:]
            num_classes = 10

            model = SigLIPVisionTransformer(
                input_shape=input_shape,
                num_classes=num_classes,
                scale="tiny",
                include_top=True
            )

            # Forward pass
            output = model(input_tensor, training=False)

            # Check output shape and properties
            assert output.shape == (batch_size, num_classes)
            assert not np.any(np.isnan(output.numpy()))
            assert not np.any(np.isinf(output.numpy()))

    def test_forward_pass_without_top(self, sample_inputs):
        """Test forward pass without classification head."""
        pooling_options = ["cls", "mean", "max", None]

        for input_name, input_tensor in sample_inputs.items():
            batch_size = input_tensor.shape[0]
            input_shape = input_tensor.shape[1:]
            embed_dim = 192  # tiny scale

            # Calculate expected sequence length
            patch_size = 16
            if input_shape[-1] == 1:  # Grayscale case
                patch_size = 8

            h_patches = input_shape[0] // patch_size
            w_patches = input_shape[1] // patch_size
            seq_len = h_patches * w_patches + 1  # +1 for CLS token

            for pooling in pooling_options:
                model = SigLIPVisionTransformer(
                    input_shape=input_shape,
                    scale="tiny",
                    include_top=False,
                    pooling=pooling,
                    patch_size=patch_size
                )

                # Forward pass
                output = model(input_tensor, training=False)

                # Check output shape based on pooling
                if pooling in ["cls", "mean", "max"]:
                    expected_shape = (batch_size, embed_dim)
                else:
                    expected_shape = (batch_size, seq_len, embed_dim)

                assert output.shape == expected_shape
                assert not np.any(np.isnan(output.numpy()))

    def test_siglip_patch_embedding_functionality(self):
        """Test that SigLIP two-stage patch embedding produces valid outputs."""
        model = SigLIPVisionTransformer(
            input_shape=(64, 64, 3),
            patch_size=16,
            scale="tiny",
            include_top=False,
            pooling=None  # Return full sequence
        )

        input_tensor = np.random.rand(2, 64, 64, 3).astype('float32')
        output = model(input_tensor, training=False)

        # Should have correct sequence length: (64/16)^2 patches + 1 CLS = 17
        expected_seq_len = 17
        expected_embed_dim = 192  # tiny scale

        assert output.shape == (2, expected_seq_len, expected_embed_dim)
        assert not np.any(np.isnan(output.numpy()))

    def test_cls_token_functionality(self):
        """Test CLS token extraction functionality."""
        model = SigLIPVisionTransformer(
            input_shape=(64, 64, 3),
            scale="tiny",
            include_top=False,
            pooling=None  # Return full sequence to test CLS extraction
        )

        input_tensor = np.random.rand(1, 64, 64, 3).astype('float32')
        full_output = model(input_tensor, training=False)

        # Test CLS token extraction
        cls_token = model.get_cls_token(full_output)
        assert cls_token.shape == (1, 192)  # tiny embed_dim
        assert not np.any(np.isnan(cls_token.numpy()))

        # CLS token should be the first token
        expected_cls = full_output[:, 0, :]
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(cls_token),
            keras.ops.convert_to_numpy(expected_cls),
            rtol=1e-6, atol=1e-6
        )

    def test_patch_tokens_functionality(self):
        """Test patch token extraction functionality."""
        model = SigLIPVisionTransformer(
            input_shape=(32, 32, 3),
            patch_size=16,
            scale="tiny",
            include_top=False,
            pooling=None
        )

        input_tensor = np.random.rand(1, 32, 32, 3).astype('float32')
        full_output = model(input_tensor, training=False)

        # Test patch token extraction
        patch_tokens = model.get_patch_tokens(full_output)
        expected_num_patches = (32 // 16) ** 2  # 4 patches

        assert patch_tokens.shape == (1, expected_num_patches, 192)
        assert not np.any(np.isnan(patch_tokens.numpy()))

        # Patch tokens should be everything except the first token
        expected_patches = full_output[:, 1:, :]
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(patch_tokens),
            keras.ops.convert_to_numpy(expected_patches),
            rtol=1e-6, atol=1e-6
        )

    def test_spatial_features_functionality(self):
        """Test spatial feature reshaping functionality."""
        model = SigLIPVisionTransformer(
            input_shape=(64, 64, 3),
            patch_size=16,
            scale="tiny",
            include_top=False,
            pooling=None
        )

        input_tensor = np.random.rand(2, 64, 64, 3).astype('float32')
        full_output = model(input_tensor, training=False)

        # Test spatial feature reshaping
        spatial_features = model.get_spatial_features(full_output)

        # Should reshape patch tokens back to spatial format
        patch_h = patch_w = 64 // 16  # 4x4 patches
        expected_shape = (2, patch_h, patch_w, 192)

        assert spatial_features.shape == expected_shape
        assert not np.any(np.isnan(spatial_features.numpy()))

    def test_training_vs_inference_mode(self):
        """Test different behavior in training vs inference mode."""
        model = SigLIPVisionTransformer(scale="tiny", dropout_rate=0.5)  # High dropout for testing
        input_tensor = np.random.rand(2, 224, 224, 3).astype('float32')

        # Get outputs in both modes
        training_output = model(input_tensor, training=True)
        inference_output = model(input_tensor, training=False)

        # Shapes should be the same
        assert training_output.shape == inference_output.shape

        # With high dropout, outputs should be different
        # (though this is stochastic, so we just check they're valid)
        assert not np.any(np.isnan(training_output.numpy()))
        assert not np.any(np.isnan(inference_output.numpy()))

    def test_deterministic_output_with_controlled_inputs(self):
        """Test deterministic output with controlled inputs."""
        # Create a simple case for testing
        model = SigLIPVisionTransformer(
            input_shape=(32, 32, 3),
            patch_size=16,
            scale="tiny",
            dropout_rate=0.0,  # No randomness
            include_top=True,
            num_classes=2
        )

        # Controlled input (all ones)
        controlled_input = np.ones((1, 32, 32, 3), dtype='float32')

        # Forward pass
        output1 = model(controlled_input, training=False)

        # Second forward pass on same model should be identical
        output2 = model(controlled_input, training=False)

        # Same model, same input should produce identical results
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(output1),
            keras.ops.convert_to_numpy(output2),
            rtol=1e-6, atol=1e-6,
            err_msg="Deterministic outputs should match"
        )

        # Test that output is valid
        assert output1.shape == (1, 2)
        assert not np.any(np.isnan(output1.numpy()))
        assert not np.any(np.isinf(output1.numpy()))


class TestSigLIPSerialization:
    """Test suite for SigLIP ViT serialization following modern Keras 3 patterns."""

    def test_get_config(self):
        """Test get_config method includes all parameters."""
        model = SigLIPVisionTransformer(
            input_shape=(384, 384, 3),
            num_classes=10,
            scale="small",
            patch_size=(32, 32),
            include_top=False,
            pooling="cls",
            dropout_rate=0.1,
            normalization_type="rms_norm",
            normalization_position="pre",
            ffn_type="swiglu"
        )

        config = model.get_config()

        # Check that all important parameters are saved
        assert config["input_shape"] == (384, 384, 3)
        assert config["num_classes"] == 10
        assert config["scale"] == "small"
        assert config["patch_size"] == (32, 32)
        assert config["include_top"] is False
        assert config["pooling"] == "cls"
        assert config["dropout_rate"] == 0.1
        assert config["normalization_type"] == "rms_norm"
        assert config["normalization_position"] == "pre"
        assert config["ffn_type"] == "swiglu"

    def test_from_config_recreation(self):
        """Test model recreation from config."""
        original_model = SigLIPVisionTransformer(
            input_shape=(224, 224, 3),
            num_classes=5,
            scale="tiny",
            dropout_rate=0.2,
            normalization_type="rms_norm"
        )

        # Get config
        config = original_model.get_config()

        # Create new model from config
        recreated_model = SigLIPVisionTransformer(**{k: v for k, v in config.items() if k != 'name'})

        # Check that parameters match
        assert recreated_model.input_shape_config == original_model.input_shape_config
        assert recreated_model.num_classes == original_model.num_classes
        assert recreated_model.scale == original_model.scale
        assert recreated_model.dropout_rate == original_model.dropout_rate
        assert recreated_model.normalization_type == original_model.normalization_type

    def test_model_save_load_cycle(self):
        """CRITICAL TEST: Complete model save/load cycle with modern serialization."""
        # Create and test a small model
        original_model = SigLIPVisionTransformer(
            input_shape=(64, 64, 3),
            num_classes=3,
            scale="tiny",
            patch_size=16,
            normalization_type="rms_norm",
            ffn_type="swiglu"
        )

        # Generate test data and prediction
        test_input = np.random.rand(2, 64, 64, 3).astype('float32')
        original_output = original_model(test_input, training=False)

        # Save and load model (modern registration handles serialization)
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, 'test_siglip_vit.keras')
            original_model.save(model_path)

            # Load model - @keras.saving.register_keras_serializable() handles this
            loaded_model = keras.models.load_model(model_path)

            # Test loaded model
            loaded_output = loaded_model(test_input, training=False)

            # Outputs should match exactly
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_output),
                keras.ops.convert_to_numpy(loaded_output),
                rtol=1e-6, atol=1e-6,
                err_msg="Loaded model predictions should match original"
            )

            # Model configurations should match
            assert loaded_model.input_shape_config == original_model.input_shape_config
            assert loaded_model.num_classes == original_model.num_classes
            assert loaded_model.scale == original_model.scale
            assert loaded_model.normalization_type == original_model.normalization_type

    def test_siglip_specific_serialization(self):
        """Test serialization of SigLIP-specific components."""
        model = SigLIPVisionTransformer(
            input_shape=(128, 128, 3),
            scale="tiny",
            patch_size=32
        )

        # Build model to create SigLIP components
        model.build((None, 128, 128, 3))

        # Test that SigLIP patch embedding structure is preserved
        config = model.get_config()

        # Create new model and build it
        new_model = SigLIPVisionTransformer(**{k: v for k, v in config.items() if k != 'name'})
        new_model.build((None, 128, 128, 3))

        # Both should have the same two-stage patch embedding structure
        assert isinstance(model.siglip_patch_embed, keras.Sequential)
        assert isinstance(new_model.siglip_patch_embed, keras.Sequential)
        assert len(model.siglip_patch_embed.layers) == len(new_model.siglip_patch_embed.layers)


class TestSigLIPFactoryFunctions:
    """Test suite for SigLIP ViT factory functions."""

    def test_create_siglip_vision_transformer(self):
        """Test the main factory function with validation."""
        model = create_siglip_vision_transformer(
            input_shape=(224, 224, 3),
            num_classes=1000,
            scale="base"
        )

        assert isinstance(model, SigLIPVisionTransformer)
        assert model.scale == "base"
        assert model.num_classes == 1000

    def test_factory_function_validation(self):
        """Test validation in factory functions."""
        # Invalid parameters should raise errors
        with pytest.raises(ValueError, match="num_classes must be positive"):
            create_siglip_vision_transformer(num_classes=-1)

        with pytest.raises(ValueError, match="input_shape must be a 3-element"):
            create_siglip_vision_transformer(input_shape=(224, 224))

        with pytest.raises(ValueError, match="patch_size must be positive"):
            create_siglip_vision_transformer(patch_size=-16)

        with pytest.raises(ValueError, match="Image height .* must be divisible"):
            create_siglip_vision_transformer(input_shape=(225, 224, 3), patch_size=16)

    def test_factory_function_with_advanced_options(self):
        """Test factory functions with advanced configuration options."""
        model = create_siglip_vision_transformer(
            input_shape=(384, 384, 3),
            num_classes=100,
            scale="small",
            patch_size=32,
            include_top=False,
            pooling="cls",
            dropout_rate=0.1,
            normalization_type="rms_norm",
            normalization_position="pre",
            ffn_type="swiglu"
        )

        assert model.input_shape_config == (384, 384, 3)
        assert model.num_classes == 100
        assert model.scale == "small"
        assert model.patch_size == (32, 32)
        assert model.include_top is False
        assert model.pooling == "cls"
        assert model.normalization_type == "rms_norm"
        assert model.ffn_type == "swiglu"


class TestSigLIPTrainingIntegration:
    """Test suite for SigLIP ViT training integration."""

    def test_model_compilation(self):
        """Test model compilation for training."""
        model = SigLIPVisionTransformer(
            input_shape=(64, 64, 3),
            num_classes=5,
            scale="tiny"
        )

        # Compile model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Check that model is compiled
        assert model.compiled_loss is not None
        assert model.optimizer is not None

    def test_simple_training_loop(self):
        """Test a simple training loop."""
        # Create small model and dataset
        model = SigLIPVisionTransformer(
            input_shape=(32, 32, 3),
            num_classes=2,
            scale="tiny",
            patch_size=16
        )

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Create synthetic data
        x_train = np.random.random((16, 32, 32, 3)).astype('float32')
        y_train = np.random.randint(0, 2, (16,)).astype('int32')

        # Train for a few steps
        history = model.fit(x_train, y_train, epochs=2, batch_size=8, verbose=0)

        # Check that training worked
        assert 'loss' in history.history
        assert 'accuracy' in history.history
        assert len(history.history['loss']) == 2

    def test_feature_extractor_creation(self):
        """Test creating feature extractor from trained model."""
        model = SigLIPVisionTransformer(
            input_shape=(64, 64, 3),
            num_classes=10,
            scale="tiny",
            include_top=True
        )

        # Build the model
        model.build((None, 64, 64, 3))

        # Create feature extractor
        feature_model = model.get_feature_extractor()

        assert feature_model.include_top is False
        assert feature_model.pooling == "cls"
        assert feature_model.input_shape_config == model.input_shape_config
        assert feature_model.scale == model.scale

    def test_model_summary_detailed(self):
        """Test detailed model summary functionality."""
        model = SigLIPVisionTransformer(
            input_shape=(224, 224, 3),
            num_classes=1000,
            scale="base"
        )

        # Build the model to get parameter counts
        model.build((None, 224, 224, 3))

        # This should not raise an error and should mention SigLIP features
        model.summary_detailed()


class TestSigLIPEdgeCases:
    """Test suite for SigLIP ViT edge cases and robustness."""

    def test_minimal_viable_configuration(self):
        """Test with minimal viable image and patch sizes."""
        # Smallest reasonable configuration
        model = SigLIPVisionTransformer(
            input_shape=(16, 16, 1),
            patch_size=8,
            scale="tiny",
            num_classes=2
        )

        input_tensor = np.random.rand(1, 16, 16, 1).astype('float32')
        output = model(input_tensor)

        assert output.shape == (1, 2)
        assert not np.any(np.isnan(output.numpy()))

    def test_large_patch_configuration(self):
        """Test with large patches (fewer patches)."""
        model = SigLIPVisionTransformer(
            input_shape=(224, 224, 3),
            patch_size=56,  # Results in 4x4 = 16 patches
            scale="tiny",
            num_classes=10
        )

        # Should have 16 patches + 1 CLS = 17 sequence length
        assert model.num_patches == 16
        assert model.max_seq_len == 17

        input_tensor = np.random.rand(2, 224, 224, 3).astype('float32')
        output = model(input_tensor)

        assert output.shape == (2, 10)

    def test_unusual_aspect_ratios(self):
        """Test with unusual aspect ratios."""
        # Wide image
        model = SigLIPVisionTransformer(
            input_shape=(128, 256, 3),
            patch_size=(16, 32),  # Matching aspect ratio patches
            scale="tiny"
        )

        expected_patches = (128 // 16) * (256 // 32)  # 8 * 8 = 64
        assert model.num_patches == expected_patches

        input_tensor = np.random.rand(1, 128, 256, 3).astype('float32')
        output = model(input_tensor)

        assert not np.any(np.isnan(output.numpy()))

    def test_different_normalization_configurations(self):
        """Test different normalization configurations using factories."""
        configs = [
            {"normalization_type": "layer_norm", "normalization_position": "post"},
            {"normalization_type": "layer_norm", "normalization_position": "pre"},
            {"normalization_type": "rms_norm", "normalization_position": "post"},
            {"normalization_type": "rms_norm", "normalization_position": "pre"}
        ]

        for config in configs:
            model = SigLIPVisionTransformer(
                input_shape=(64, 64, 3),
                scale="tiny",
                num_classes=5,
                **config
            )

            input_tensor = np.random.rand(2, 64, 64, 3).astype('float32')
            output = model(input_tensor)

            assert output.shape == (2, 5)
            assert not np.any(np.isnan(output.numpy()))

    def test_different_ffn_types(self):
        """Test different FFN types using factory integration."""
        ffn_types = ["mlp", "swiglu", "geglu"]  # Available types

        for ffn_type in ffn_types:
            model = SigLIPVisionTransformer(
                input_shape=(64, 64, 3),
                scale="tiny",
                num_classes=3,
                ffn_type=ffn_type
            )

            input_tensor = np.random.rand(1, 64, 64, 3).astype('float32')
            output = model(input_tensor)

            assert output.shape == (1, 3)
            assert not np.any(np.isnan(output.numpy()))

    def test_extreme_dropout_rates(self):
        """Test with extreme dropout rates."""
        # Very high dropout (should still work in inference mode)
        model = SigLIPVisionTransformer(
            input_shape=(64, 64, 3),
            scale="tiny",
            dropout_rate=0.9,
            attention_dropout_rate=0.8,
            pos_dropout_rate=0.7
        )

        input_tensor = np.random.rand(1, 64, 64, 3).astype('float32')

        # Should work fine in inference mode
        output = model(input_tensor, training=False)
        assert not np.any(np.isnan(output.numpy()))

    def test_numerical_stability_with_extreme_inputs(self):
        """Test numerical stability with extreme input values."""
        model = SigLIPVisionTransformer(input_shape=(64, 64, 3), scale="tiny", num_classes=2)

        test_cases = [
            np.zeros((1, 64, 64, 3), dtype='float32'),  # All zeros
            np.ones((1, 64, 64, 3), dtype='float32') * 1e-10,  # Very small values
            np.ones((1, 64, 64, 3), dtype='float32') * 100,  # Large values
        ]

        for test_input in test_cases:
            output = model(test_input, training=False)

            # Check for numerical issues
            assert not np.any(np.isnan(output.numpy())), "NaN values detected"
            assert not np.any(np.isinf(output.numpy())), "Inf values detected"


class TestSigLIPRegularizationIntegration:
    """Test suite for SigLIP ViT with regularization."""

    def test_with_regularizers(self):
        """Test SigLIP ViT with kernel and bias regularizers."""
        model = SigLIPVisionTransformer(
            input_shape=(64, 64, 3),
            scale="tiny",
            num_classes=5,
            kernel_regularizer=keras.regularizers.L2(0.01),
            bias_regularizer=keras.regularizers.L1(0.01)
        )

        # Build the model to create regularization losses
        input_tensor = np.random.rand(2, 64, 64, 3).astype('float32')
        output = model(input_tensor)

        # Should have regularization losses
        assert len(model.losses) > 0

        # Output should still be valid
        assert output.shape == (2, 5)
        assert not np.any(np.isnan(output.numpy()))

    def test_regularizer_serialization(self):
        """Test that regularizers are properly serialized."""
        original_model = SigLIPVisionTransformer(
            input_shape=(32, 32, 3),
            scale="tiny",
            num_classes=2,
            kernel_regularizer=keras.regularizers.L2(0.01)
        )

        # Get config and verify regularizer is serialized
        config = original_model.get_config()

        # Regularizer should be serialized in config
        kernel_reg_config = config['kernel_regularizer']
        assert kernel_reg_config is not None


class TestSigLIPArchitectureValidation:
    """Test suite for validating the overall SigLIP ViT architecture."""

    def test_transformer_layer_integration(self):
        """Test that TransformerLayer integration works correctly."""
        model = SigLIPVisionTransformer(
            input_shape=(64, 64, 3),
            scale="tiny",
            num_classes=5,
            normalization_type="rms_norm",
            ffn_type="swiglu"
        )

        # Build model to create transformer layers
        model.build((None, 64, 64, 3))

        # Verify transformer layers are created correctly
        assert len(model.transformer_layers) == model.num_layers
        for layer in model.transformer_layers:
            # Check that transformer layers have correct configuration
            assert layer.hidden_size == model.embed_dim
            assert layer.num_heads == model.num_heads
            assert layer.normalization_type == "rms_norm"
            assert layer.ffn_type == "swiglu"

    def test_factory_component_integration(self):
        """Test that factory-created components integrate properly."""
        model = SigLIPVisionTransformer(
            input_shape=(128, 128, 3),
            patch_size=16,
            scale="small",
            normalization_type="rms_norm"
        )

        # Build model
        model.build((None, 128, 128, 3))

        # Test that SigLIP patch embedding is created properly
        assert model.siglip_patch_embed is not None
        assert isinstance(model.siglip_patch_embed, keras.Sequential)

        # Test that positional embedding was created using factory
        assert model.pos_embed is not None
        assert hasattr(model.pos_embed, 'max_seq_len')

        # Test that normalization handling is correct (pre vs post)
        if model.normalization_position == "pre":
            assert model.norm is not None
        else:
            assert model.norm is None

    def test_siglip_vs_standard_vit_differences(self):
        """Test that SigLIP has architectural differences from standard ViT."""
        model = SigLIPVisionTransformer(input_shape=(64, 64, 3), scale="tiny")
        model.build((None, 64, 64, 3))

        # SigLIP should have two-stage patch embedding (Sequential with 4 layers)
        assert isinstance(model.siglip_patch_embed, keras.Sequential)
        assert len(model.siglip_patch_embed.layers) == 4

        # First stage should produce embed_dim // 2 features
        first_conv = model.siglip_patch_embed.layers[0]
        assert isinstance(first_conv, keras.layers.Conv2D)
        assert first_conv.filters == model.embed_dim // 2

        # Second stage should produce full embed_dim features
        second_conv = model.siglip_patch_embed.layers[3]
        assert isinstance(second_conv, keras.layers.Conv2D)
        assert second_conv.filters == model.embed_dim

    def test_end_to_end_functionality(self):
        """Test complete end-to-end functionality."""
        # Create model with various configurations
        model = SigLIPVisionTransformer(
            input_shape=(96, 96, 3),
            num_classes=7,
            scale="tiny",
            patch_size=12,
            include_top=True,
            dropout_rate=0.1,
            attention_dropout_rate=0.1,
            normalization_type="rms_norm",
            normalization_position="pre",
            ffn_type="swiglu"
        )

        # Test input
        test_input = np.random.rand(3, 96, 96, 3).astype('float32')

        # Forward pass
        output = model(test_input, training=True)

        # Validate output
        assert output.shape == (3, 7)
        assert not np.any(np.isnan(output.numpy()))

        # Test that model can be saved and loaded
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, 'end_to_end_siglip_test.keras')
            model.save(model_path)

            loaded_model = keras.models.load_model(model_path)
            loaded_output = loaded_model(test_input, training=False)

            # Outputs should be valid (though may differ due to training mode)
            assert loaded_output.shape == output.shape
            assert not np.any(np.isnan(loaded_output.numpy()))

    def test_multimodal_readiness(self):
        """Test that SigLIP is properly configured for multimodal training."""
        # SigLIP is designed for multimodal learning, so test feature extraction capabilities
        model = SigLIPVisionTransformer(
            input_shape=(224, 224, 3),
            scale="base",
            include_top=False,
            pooling="cls"  # CLS token for multimodal fusion
        )

        # Test that feature extraction works properly
        input_tensor = np.random.rand(4, 224, 224, 3).astype('float32')
        features = model(input_tensor, training=False)

        # Should produce feature vectors suitable for multimodal fusion
        assert features.shape == (4, model.embed_dim)
        assert not np.any(np.isnan(features.numpy()))

        # Test that spatial features can be extracted for dense tasks
        full_features = SigLIPVisionTransformer(
            input_shape=(224, 224, 3),
            scale="base",
            include_top=False,
            pooling=None
        )(input_tensor)

        spatial_features = model.get_spatial_features(full_features)
        expected_spatial_shape = (4, 14, 14, model.embed_dim)  # 224/16 = 14
        assert spatial_features.shape == expected_spatial_shape


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])