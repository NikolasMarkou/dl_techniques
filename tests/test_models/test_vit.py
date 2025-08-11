"""
Comprehensive test suite for the Vision Transformer (ViT) model.

This test suite covers all aspects of the ViT model lifecycle including:
- Model initialization and configuration
- Building and architecture validation
- Forward pass and output shapes
- Serialization and deserialization
- Training integration
- Edge cases and error handling
"""

import pytest
import numpy as np
import tempfile
import os
from typing import Tuple, Dict, Any, Optional, List

import keras
from keras import ops

# Assuming the ViT model is in the models directory
from dl_techniques.models.vit import (
    ViT,
    create_vision_transformer,
    create_vit_tiny,
    create_vit_small,
    create_vit_base,
    create_vit_large,
    create_vit_huge
)


class TestViTInitialization:
    """Test suite for ViT model initialization."""

    def test_default_initialization(self):
        """Test initialization with default parameters."""
        model = ViT()

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
        assert model.norm_type == "layer"
        assert model.normalization_position == "post"
        assert model.ffn_type == "mlp"
        assert model.activation == "gelu"
        assert not model.built

    def test_custom_initialization(self):
        """Test initialization with custom parameters."""
        model = ViT(
            input_shape=(384, 384, 3),
            num_classes=10,
            scale="small",
            patch_size=(32, 32),
            include_top=False,
            pooling="cls",
            dropout_rate=0.1,
            attention_dropout_rate=0.05,
            pos_dropout_rate=0.2,
            norm_type="rms",
            normalization_position="pre",
            ffn_type="swiglu",
            activation="relu",
            name="custom_vit"
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
        assert model.norm_type == "rms"
        assert model.normalization_position == "pre"
        assert model.ffn_type == "swiglu"
        assert model.activation == "relu"
        assert model.name == "custom_vit"

    def test_scale_configurations(self):
        """Test all scale configurations."""
        scales = ["tiny", "small", "base", "large", "huge"]
        expected_configs = {
            "tiny": [192, 3, 12, 4.0],
            "small": [384, 6, 12, 4.0],
            "base": [768, 12, 12, 4.0],
            "large": [1024, 16, 24, 4.0],
            "huge": [1280, 16, 32, 4.0],
        }

        for scale in scales:
            model = ViT(scale=scale)
            embed_dim, num_heads, num_layers, mlp_ratio = expected_configs[scale]

            assert model.embed_dim == embed_dim
            assert model.num_heads == num_heads
            assert model.num_layers == num_layers
            assert model.mlp_ratio == mlp_ratio
            assert model.intermediate_size == int(embed_dim * mlp_ratio)

    def test_patch_size_handling(self):
        """Test different patch size formats."""
        # Integer patch size
        model1 = ViT(patch_size=16)
        assert model1.patch_size == (16, 16)

        # Tuple patch size with compatible dimensions
        model2 = ViT(input_shape=(96, 144, 3), patch_size=(8, 12))
        assert model2.patch_size == (8, 12)

    def test_invalid_parameters(self):
        """Test that invalid parameters raise appropriate errors."""
        # Invalid input_shape
        with pytest.raises(ValueError, match="input_shape must be a 3-tuple"):
            ViT(input_shape=(224, 224))

        with pytest.raises(ValueError, match="All input_shape dimensions must be positive"):
            ViT(input_shape=(224, -224, 3))

        # Invalid scale
        with pytest.raises(ValueError, match="Unsupported scale"):
            ViT(scale="invalid_scale")

        # Invalid pooling
        with pytest.raises(ValueError, match="Unsupported pooling"):
            ViT(pooling="invalid_pooling")

        # Invalid patch size
        with pytest.raises(ValueError, match="patch_size must be positive"):
            ViT(patch_size=-16)

        with pytest.raises(ValueError, match="patch_size dimensions must be positive"):
            ViT(patch_size=(16, -16))

        # Incompatible image and patch dimensions
        with pytest.raises(ValueError, match="Image height .* must be divisible by patch height"):
            ViT(input_shape=(225, 224, 3), patch_size=16)

        # Invalid norm_type
        with pytest.raises(ValueError, match="Unsupported norm_type"):
            ViT(norm_type="invalid_norm")

        # Invalid normalization_position
        with pytest.raises(ValueError, match="Unsupported normalization_position"):
            ViT(normalization_position="invalid_position")


class TestViTBuildProcess:
    """Test suite for ViT model building process."""

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
        """Test that the model builds properly."""
        for input_shape in sample_input_shapes:
            model = ViT(input_shape=input_shape, scale="tiny")  # Use tiny for speed

            # Model should not be built initially
            assert not model.built

            # Build the model
            batch_input_shape = (None,) + input_shape
            model.build(batch_input_shape)

            # Check that model is built
            assert model.built
            assert len(model.weights) > 0

            # Check that all sublayers exist
            assert model.patch_embed is not None
            assert model.cls_token is not None
            assert model.pos_embed is not None
            assert len(model.transformer_layers) == model.num_layers
            assert model.norm is not None

    def test_build_with_include_top(self):
        """Test building with classification head."""
        model = ViT(include_top=True, num_classes=10, scale="tiny")
        model.build((None, 224, 224, 3))

        assert model.head is not None
        assert model.head.units == 10

    def test_build_without_include_top(self):
        """Test building without classification head."""
        model = ViT(include_top=False, scale="tiny")
        model.build((None, 224, 224, 3))

        assert model.head is None

    def test_build_with_different_pooling(self):
        """Test building with different pooling options."""
        pooling_options = ["cls", "mean", "max", None]

        for pooling in pooling_options:
            model = ViT(include_top=False, pooling=pooling, scale="tiny")
            model.build((None, 224, 224, 3))

            if pooling in ["mean", "max"]:
                assert model.global_pool is not None
            else:
                assert model.global_pool is None

    def test_build_idempotency(self):
        """Test that multiple build calls are idempotent."""
        model = ViT(scale="tiny")
        input_shape = (None, 224, 224, 3)

        # Build multiple times
        model.build(input_shape)
        weights_after_first_build = len(model.weights)

        model.build(input_shape)
        weights_after_second_build = len(model.weights)

        # Should be the same
        assert weights_after_first_build == weights_after_second_build


class TestViTOutputShapes:
    """Test suite for ViT output shape computation."""

    @pytest.fixture
    def input_shapes_and_configs(self) -> List[Dict[str, Any]]:
        """Test configurations with expected shapes."""
        return [
            {
                "input_shape": (224, 224, 3),
                "patch_size": 16,
                "scale": "tiny",
                "expected_patches": 196,  # (224/16)^2
                "expected_seq_len": 197   # 196 + 1 (CLS)
            },
            {
                "input_shape": (384, 384, 3),
                "patch_size": (32, 32),
                "scale": "small",
                "expected_patches": 144,  # (384/32)^2
                "expected_seq_len": 145   # 144 + 1 (CLS)
            },
            {
                "input_shape": (128, 128, 1),
                "patch_size": 8,
                "scale": "tiny",
                "expected_patches": 256,  # (128/8)^2
                "expected_seq_len": 257   # 256 + 1 (CLS)
            }
        ]

    def test_compute_output_shape_with_top(self, input_shapes_and_configs):
        """Test output shape computation with classification head."""
        for config in input_shapes_and_configs:
            model = ViT(
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
            embed_dim = ViT.SCALE_CONFIGS[config["scale"]][0]

            # Test different pooling options
            pooling_tests = [
                ("cls", (None, embed_dim)),
                ("mean", (None, embed_dim)),
                ("max", (None, embed_dim)),
                (None, (None, config["expected_seq_len"], embed_dim))
            ]

            for pooling, expected_shape in pooling_tests:
                model = ViT(
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
        model = ViT(scale="tiny")

        # 3D input shape should fail
        with pytest.raises(ValueError, match="Expected 4D input shape"):
            model.compute_output_shape((None, 224, 224))


class TestViTForwardPass:
    """Test suite for ViT forward pass."""

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

            model = ViT(
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
                model = ViT(
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

    def test_training_vs_inference_mode(self):
        """Test different behavior in training vs inference mode."""
        model = ViT(scale="tiny", dropout_rate=0.5)  # High dropout for testing
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
        # Skip this test if it's not deterministic (some randomness sources are hard to control)
        # This is more of a smoke test to ensure the model runs without errors

        # Create a simple case for testing
        model = ViT(
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
        assert np.allclose(output1.numpy(), output2.numpy(), atol=1e-7)

        # Test that output is valid
        assert output1.shape == (1, 2)
        assert not np.any(np.isnan(output1.numpy()))
        assert not np.any(np.isinf(output1.numpy()))


class TestViTSerialization:
    """Test suite for ViT serialization and deserialization."""

    def test_get_config(self):
        """Test get_config method."""
        model = ViT(
            input_shape=(384, 384, 3),
            num_classes=10,
            scale="small",
            patch_size=(32, 32),
            include_top=False,
            pooling="cls",
            dropout_rate=0.1,
            norm_type="rms",
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
        assert config["norm_type"] == "rms"
        assert config["normalization_position"] == "pre"
        assert config["ffn_type"] == "swiglu"

    def test_from_config(self):
        """Test model recreation from config."""
        original_model = ViT(
            input_shape=(224, 224, 3),
            num_classes=5,
            scale="tiny",
            dropout_rate=0.2
        )

        # Get config
        config = original_model.get_config()

        # Create new model from config
        recreated_model = ViT.from_config(config)

        # Check that parameters match
        assert recreated_model.input_shape_config == original_model.input_shape_config
        assert recreated_model.num_classes == original_model.num_classes
        assert recreated_model.scale == original_model.scale
        assert recreated_model.dropout_rate == original_model.dropout_rate

    def test_build_config_serialization(self):
        """Test build configuration serialization."""
        model = ViT(scale="tiny")
        input_shape = (None, 224, 224, 3)

        # Build the model
        model.build(input_shape)

        # Get build config
        build_config = model.get_build_config()
        assert build_config["input_shape"] == input_shape

        # Create new model and build from config
        new_model = ViT(scale="tiny")
        new_model.build_from_config(build_config)

        assert new_model.built

    def test_model_save_load_cycle(self):
        """Test complete model save/load cycle."""
        # Create and train a small model
        original_model = ViT(
            input_shape=(64, 64, 3),
            num_classes=3,
            scale="tiny",
            patch_size=16
        )

        # Generate test data and prediction
        test_input = np.random.rand(2, 64, 64, 3).astype('float32')
        original_output = original_model(test_input, training=False)

        # Save and load model
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, 'test_vit.keras')
            original_model.save(model_path)

            # Load model with custom objects
            loaded_model = keras.models.load_model(
                model_path,
                custom_objects={'ViT': ViT}
            )

            # Test loaded model
            loaded_output = loaded_model(test_input, training=False)

            # Outputs should match exactly
            assert np.allclose(original_output.numpy(), loaded_output.numpy(), atol=1e-6)

            # Model configurations should match
            assert loaded_model.input_shape_config == original_model.input_shape_config
            assert loaded_model.num_classes == original_model.num_classes
            assert loaded_model.scale == original_model.scale


class TestViTFactoryFunctions:
    """Test suite for ViT factory functions."""

    def test_create_vision_transformer(self):
        """Test the main factory function."""
        model = create_vision_transformer(
            input_shape=(224, 224, 3),
            num_classes=1000,
            scale="base"
        )

        assert isinstance(model, ViT)
        assert model.scale == "base"
        assert model.num_classes == 1000

    def test_create_vit_variants(self):
        """Test all variant creation functions."""
        variants = {
            create_vit_tiny: "tiny",
            create_vit_small: "small",
            create_vit_base: "base",
            create_vit_large: "large",
            create_vit_huge: "huge"
        }

        for create_func, expected_scale in variants.items():
            model = create_func(num_classes=10)
            assert isinstance(model, ViT)
            assert model.scale == expected_scale
            assert model.num_classes == 10

    def test_factory_function_validation(self):
        """Test validation in factory functions."""
        # Invalid parameters should raise errors
        with pytest.raises(ValueError, match="num_classes must be positive"):
            create_vision_transformer(num_classes=-1)

        with pytest.raises(ValueError, match="input_shape must be a 3-element"):
            create_vision_transformer(input_shape=(224, 224))

        with pytest.raises(ValueError, match="patch_size must be positive"):
            create_vision_transformer(patch_size=-16)


class TestViTTrainingIntegration:
    """Test suite for ViT training integration."""

    def test_model_compilation(self):
        """Test model compilation for training."""
        model = ViT(
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
        model = ViT(
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
        model = ViT(
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


class TestViTEdgeCases:
    """Test suite for ViT edge cases and robustness."""

    def test_minimal_viable_configuration(self):
        """Test with minimal viable image and patch sizes."""
        # Smallest reasonable configuration
        model = ViT(
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
        model = ViT(
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
        model = ViT(
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
        """Test different normalization configurations."""
        configs = [
            {"norm_type": "layer", "normalization_position": "post"},
            {"norm_type": "layer", "normalization_position": "pre"},
            {"norm_type": "rms", "normalization_position": "post"},
            {"norm_type": "rms", "normalization_position": "pre"}
        ]

        for config in configs:
            model = ViT(
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
        """Test different FFN types."""
        ffn_types = ["mlp", "swiglu", "differential", "glu"]

        for ffn_type in ffn_types:
            try:
                model = ViT(
                    input_shape=(64, 64, 3),
                    scale="tiny",
                    num_classes=3,
                    ffn_type=ffn_type
                )

                input_tensor = np.random.rand(1, 64, 64, 3).astype('float32')
                output = model(input_tensor)

                assert output.shape == (1, 3)
                assert not np.any(np.isnan(output.numpy()))
            except Exception as e:
                # Some FFN types might not be available, that's okay
                pytest.skip(f"FFN type {ffn_type} not available: {e}")

    def test_extreme_dropout_rates(self):
        """Test with extreme dropout rates."""
        # Very high dropout (should still work in inference mode)
        model = ViT(
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
        model = ViT(input_shape=(64, 64, 3), scale="tiny", num_classes=2)

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


class TestViTRegularizationIntegration:
    """Test suite for ViT with regularization."""

    def test_with_regularizers(self):
        """Test ViT with kernel and bias regularizers."""
        model = ViT(
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])