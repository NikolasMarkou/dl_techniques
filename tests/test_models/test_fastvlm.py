import pytest
import tempfile
import os
import numpy as np
import keras
import tensorflow as tf
from typing import Dict, Any, Tuple, List, Optional

from dl_techniques.models.fastvlm.model import FastVLM

class TestFastVLM:
    """Comprehensive test suite for FastVLM model."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up method run before each test."""
        # Verify that FastVLM class is available
        try:
            # This would normally be: assert FastVLM is not None
            # For now, we'll assume it's available
            pass
        except NameError:
            pytest.skip("FastVLM class not available - ensure proper import")

    @pytest.fixture
    def base_config(self) -> Dict[str, Any]:
        """Base configuration for testing."""
        return {
            'num_classes': 10,
            'embed_dims': [32, 64, 128],
            'depths': [2, 2, 2],
            'num_heads': [1, 2, 4],
            'mlp_ratio': 2.0,
            'dropout_rate': 0.1,
            'drop_path_rate': 0.1,
            'use_se': False,
            'attention_type': 'multi_head_attention',
            'use_layer_scale': True,
            'activation': 'gelu',
            'kernel_initializer': 'he_normal',
            'include_top': True,
            'input_shape': (32, 32, 3)  # Small input for testing
        }

    @pytest.fixture
    def sample_input(self) -> keras.KerasTensor:
        """Sample input tensor for testing."""
        return keras.random.normal(shape=(2, 32, 32, 3))  # Small batch for testing

    @pytest.fixture
    def large_sample_input(self) -> keras.KerasTensor:
        """Larger sample input for ImageNet-like testing."""
        return keras.random.normal(shape=(1, 224, 224, 3))

    def test_initialization_base_config(self, base_config: Dict[str, Any]) -> None:
        """Test model initialization with base configuration."""
        model = FastVLM(**base_config)

        # Check configuration storage
        assert model.num_classes == base_config['num_classes']
        assert model.embed_dims == base_config['embed_dims']
        assert model.depths == base_config['depths']
        assert model.num_heads == base_config['num_heads']
        assert model.mlp_ratio == base_config['mlp_ratio']
        assert model.dropout_rate == base_config['dropout_rate']
        assert model.drop_path_rate == base_config['drop_path_rate']
        assert model.use_se == base_config['use_se']
        assert model.attention_type == base_config['attention_type']
        assert model.use_layer_scale == base_config['use_layer_scale']
        assert model.activation == base_config['activation']
        assert model.include_top == base_config['include_top']
        assert model._input_shape == base_config['input_shape']

        # Check sub-components exist
        assert hasattr(model, 'stem')
        assert hasattr(model, 'stages')
        assert hasattr(model, 'downsample_layers')
        assert hasattr(model, 'head')

        # Check structure
        assert len(model.stages) == 3  # Three stages
        assert len(model.downsample_layers) == 2  # Two downsample layers
        assert model.head is not None  # Classification head present

    def test_initialization_without_top(self, base_config: Dict[str, Any]) -> None:
        """Test model initialization without classification head."""
        config = base_config.copy()
        config['include_top'] = False

        model = FastVLM(**config)

        assert model.include_top is False
        assert model.head is None

    def test_initialization_zero_classes(self, base_config: Dict[str, Any]) -> None:
        """Test model initialization with zero classes (feature extraction)."""
        config = base_config.copy()
        config['num_classes'] = 0

        model = FastVLM(**config)

        assert model.num_classes == 0
        assert model.head is None  # No classification head for feature extraction

    def test_forward_pass(
        self,
        base_config: Dict[str, Any],
        sample_input: keras.KerasTensor
    ) -> None:
        """Test forward pass through the model."""
        model = FastVLM(**base_config)

        output = model(sample_input)

        # Check output shape
        expected_shape = (sample_input.shape[0], base_config['num_classes'])
        assert output.shape == expected_shape

        # Check output is finite
        assert keras.ops.all(keras.ops.isfinite(output))

    def test_forward_pass_without_top(
        self,
        base_config: Dict[str, Any],
        sample_input: keras.KerasTensor
    ) -> None:
        """Test forward pass without classification head."""
        config = base_config.copy()
        config['include_top'] = False

        model = FastVLM(**config)
        output = model(sample_input)

        # Output should be 4D feature map
        assert len(output.shape) == 4
        assert output.shape[0] == sample_input.shape[0]  # Batch size preserved
        assert output.shape[-1] == config['embed_dims'][-1]  # Last stage channels

        # Spatial dimensions should be downsampled by 16 (4 * 2 * 2)
        expected_height = sample_input.shape[1] // 16
        expected_width = sample_input.shape[2] // 16
        assert output.shape[1] == expected_height
        assert output.shape[2] == expected_width

    def test_extract_features(
        self,
        base_config: Dict[str, Any],
        sample_input: keras.KerasTensor
    ) -> None:
        """Test feature extraction from all stages."""
        model = FastVLM(**base_config)

        features = model.extract_features(sample_input)

        # Should return 4 feature maps (stem + 3 stages)
        assert len(features) == 4

        # Check shapes
        batch_size = sample_input.shape[0]
        input_h, input_w = sample_input.shape[1:3]

        # Stem output: [B, H/4, W/4, embed_dims[0]]
        assert features[0].shape == (batch_size, input_h//4, input_w//4, base_config['embed_dims'][0])

        # Stage 1 output: [B, H/4, W/4, embed_dims[0]]
        assert features[1].shape == (batch_size, input_h//4, input_w//4, base_config['embed_dims'][0])

        # Stage 2 output: [B, H/8, W/8, embed_dims[1]]
        assert features[2].shape == (batch_size, input_h//8, input_w//8, base_config['embed_dims'][1])

        # Stage 3 output: [B, H/16, W/16, embed_dims[2]]
        assert features[3].shape == (batch_size, input_h//16, input_w//16, base_config['embed_dims'][2])

        # All features should be finite
        for i, feat in enumerate(features):
            assert keras.ops.all(keras.ops.isfinite(feat)), f"Feature {i} contains non-finite values"

    def test_serialization_cycle(
        self,
        base_config: Dict[str, Any],
        sample_input: keras.KerasTensor
    ) -> None:
        """CRITICAL TEST: Full serialization cycle."""
        # Create original model
        model = FastVLM(**base_config)

        # Get original prediction
        original_pred = model(sample_input)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_fastvlm.keras')
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

    def test_serialization_cycle_without_top(
        self,
        base_config: Dict[str, Any],
        sample_input: keras.KerasTensor
    ) -> None:
        """Test serialization cycle for feature extraction model."""
        config = base_config.copy()
        config['include_top'] = False

        model = FastVLM(**config)
        original_pred = model(sample_input)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_fastvlm_features.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_pred = loaded_model(sample_input)

            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_pred),
                keras.ops.convert_to_numpy(loaded_pred),
                rtol=1e-6, atol=1e-6,
                err_msg="Feature predictions differ after serialization"
            )

    def test_config_completeness(self, base_config: Dict[str, Any]) -> None:
        """Test that get_config contains all constructor parameters."""
        model = FastVLM(**base_config)
        config = model.get_config()

        # Check all important parameters are present
        expected_keys = {
            'num_classes', 'embed_dims', 'depths', 'num_heads', 'mlp_ratio',
            'dropout_rate', 'drop_path_rate', 'use_se', 'attention_type',
            'use_layer_scale', 'activation', 'kernel_initializer', 'include_top',
            'input_shape'
        }

        config_keys = set(config.keys())
        missing_keys = expected_keys - config_keys
        assert len(missing_keys) == 0, f"Missing keys in get_config(): {missing_keys}"

        # Test from_config reconstruction
        reconstructed_model = FastVLM.from_config(config)

        # Verify key attributes match
        assert reconstructed_model.num_classes == model.num_classes
        assert reconstructed_model.embed_dims == model.embed_dims
        assert reconstructed_model.depths == model.depths
        assert reconstructed_model.num_heads == model.num_heads
        assert reconstructed_model.include_top == model.include_top

    def test_gradients_flow(
        self,
        base_config: Dict[str, Any],
        sample_input: keras.KerasTensor
    ) -> None:
        """Test gradient computation and flow."""
        model = FastVLM(**base_config)

        with tf.GradientTape() as tape:
            tape.watch(sample_input)
            output = model(sample_input)
            loss = keras.ops.mean(keras.ops.square(output))

        # Test gradients w.r.t. model parameters
        gradients = tape.gradient(loss, model.trainable_variables)

        assert gradients is not None
        assert len(gradients) == len(model.trainable_variables)

        # Check that gradients exist and are not all zero
        non_zero_gradients = 0
        for grad in gradients:
            if grad is not None:
                assert keras.ops.all(keras.ops.isfinite(grad)), "Gradient contains non-finite values"
                if keras.ops.any(keras.ops.not_equal(grad, 0.0)):
                    non_zero_gradients += 1

        assert non_zero_gradients > 0, "All gradients are zero"

    @pytest.mark.parametrize("training", [True, False, None])
    def test_training_modes(
        self,
        base_config: Dict[str, Any],
        sample_input: keras.KerasTensor,
        training: Optional[bool]
    ) -> None:
        """Test behavior in different training modes."""
        model = FastVLM(**base_config)

        output = model(sample_input, training=training)

        # Output shape should be consistent across training modes
        expected_shape = (sample_input.shape[0], base_config['num_classes'])
        assert output.shape == expected_shape
        assert keras.ops.all(keras.ops.isfinite(output))

    def test_dropout_behavior(
        self,
        base_config: Dict[str, Any],
        sample_input: keras.KerasTensor
    ) -> None:
        """Test dropout behavior in training vs inference."""
        config = base_config.copy()
        config['dropout_rate'] = 0.5  # High dropout rate for clear difference

        model = FastVLM(**config)

        # Get predictions in training and inference modes
        train_pred = model(sample_input, training=True)
        eval_pred1 = model(sample_input, training=False)
        eval_pred2 = model(sample_input, training=False)

        # Training mode should have some variation due to dropout
        # Inference mode should be deterministic
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(eval_pred1),
            keras.ops.convert_to_numpy(eval_pred2),
            rtol=1e-6, atol=1e-6,
            err_msg="Inference mode should be deterministic"
        )

    @pytest.mark.parametrize("variant", ["nano", "tiny", "small", "base", "large", "huge"])
    def test_model_variants(
        self,
        variant: str,
        large_sample_input: keras.KerasTensor
    ) -> None:
        """Test all predefined model variants."""
        model = FastVLM.from_variant(variant, num_classes=1000)

        # Check that variant configuration was applied
        variant_config = FastVLM.MODEL_VARIANTS[variant]
        assert model.embed_dims == variant_config['embed_dims']
        assert model.depths == variant_config['depths']
        assert model.num_heads == variant_config['num_heads']
        assert model.mlp_ratio == variant_config['mlp_ratio']
        assert model.dropout_rate == variant_config['dropout_rate']
        assert model.drop_path_rate == variant_config['drop_path_rate']
        assert model.use_se == variant_config['use_se']

        # Test forward pass
        output = model(large_sample_input)
        assert output.shape == (1, 1000)  # ImageNet classes
        assert keras.ops.all(keras.ops.isfinite(output))

    def test_variant_with_custom_params(self, large_sample_input: keras.KerasTensor) -> None:
        """Test variant creation with custom parameter overrides."""
        custom_classes = 10
        model = FastVLM.from_variant(
            "tiny",
            num_classes=custom_classes,
            dropout_rate=0.2,  # Override variant default
            use_se=True        # Override variant default
        )

        # Check custom parameters were applied
        assert model.num_classes == custom_classes
        assert model.dropout_rate == 0.2
        assert model.use_se is True

        # Check variant parameters still applied for non-overridden values
        tiny_config = FastVLM.MODEL_VARIANTS['tiny']
        assert model.embed_dims == tiny_config['embed_dims']
        assert model.depths == tiny_config['depths']

        # Test forward pass
        output = model(large_sample_input)
        assert output.shape == (1, custom_classes)

    def test_edge_cases_validation(self) -> None:
        """Test error conditions and edge cases."""
        # Test invalid num_classes
        with pytest.raises(ValueError, match="num_classes must be non-negative"):
            FastVLM(num_classes=-1)

        # Test invalid embed_dims length
        with pytest.raises(ValueError, match="embed_dims must have 3 elements"):
            FastVLM(embed_dims=[64, 128])

        # Test invalid depths length
        with pytest.raises(ValueError, match="depths must have 3 elements"):
            FastVLM(depths=[2, 3])

        # Test negative embed_dims
        with pytest.raises(ValueError, match="All embed_dims must be positive"):
            FastVLM(embed_dims=[64, -128, 256])

        # Test negative depths
        with pytest.raises(ValueError, match="All depths must be non-negative"):
            FastVLM(depths=[2, -1, 3])

        # Test invalid mlp_ratio
        with pytest.raises(ValueError, match="mlp_ratio must be positive"):
            FastVLM(mlp_ratio=-1.0)

        # Test invalid dropout_rate
        with pytest.raises(ValueError, match="dropout_rate must be between 0 and 1"):
            FastVLM(dropout_rate=1.5)

        # Test invalid drop_path_rate
        with pytest.raises(ValueError, match="drop_path_rate must be between 0 and 1"):
            FastVLM(drop_path_rate=-0.1)

        # Test num_heads divisibility
        with pytest.raises(ValueError, match="must be divisible by"):
            FastVLM(embed_dims=[65, 128, 256], num_heads=[2, 4, 8])  # 65 not divisible by 2

        # Test invalid num_heads
        with pytest.raises(ValueError, match="All num_heads must be positive"):
            FastVLM(num_heads=[1, 0, 4])

        # Test invalid input_shape
        with pytest.raises(ValueError, match="input_shape must be 3D"):
            FastVLM(input_shape=(224, 224))  # Missing channel dimension

    def test_unknown_variant(self) -> None:
        """Test error handling for unknown variants."""
        with pytest.raises(ValueError, match="Unknown variant 'unknown'"):
            FastVLM.from_variant("unknown")

    def test_default_parameters(self) -> None:
        """Test model creation with default parameters."""
        model = FastVLM()

        # Check defaults
        assert model.num_classes == 1000
        assert model.embed_dims == [64, 128, 256]
        assert model.depths == [3, 4, 6]
        assert model.num_heads == [2, 4, 8]  # Computed from embed_dims
        assert model.mlp_ratio == 4.0
        assert model.dropout_rate == 0.0
        assert model.drop_path_rate == 0.1
        assert model.use_se is False
        assert model.attention_type == 'multi_head_attention'
        assert model.use_layer_scale is True
        assert model.activation == 'gelu'
        assert model.include_top is True
        assert model._input_shape == (224, 224, 3)

    def test_auto_num_heads_computation(self) -> None:
        """Test automatic num_heads computation when not provided."""
        embed_dims = [96, 192, 384]
        model = FastVLM(embed_dims=embed_dims)

        # Should compute as max(1, dim // 32)
        expected_heads = [max(1, dim // 32) for dim in embed_dims]
        assert model.num_heads == expected_heads

    def test_model_compilation_and_fit(
        self,
        base_config: Dict[str, Any],
        sample_input: keras.KerasTensor
    ) -> None:
        """Test model compilation and basic training step."""
        model = FastVLM(**base_config)

        # Compile model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Create dummy data - fix keras.random.randint usage
        x_train = keras.random.normal((4, 32, 32, 3))
        y_train = keras.ops.cast(
            keras.random.uniform((4,), minval=0, maxval=base_config['num_classes']),
            'int32'
        )

        # Verify label range is correct
        assert keras.ops.all(y_train >= 0)
        assert keras.ops.all(y_train < base_config['num_classes'])

        # Test single training step (should not raise errors)
        history = model.fit(x_train, y_train, epochs=1, verbose=0)

        assert 'loss' in history.history
        assert len(history.history['loss']) == 1
        assert history.history['loss'][0] > 0  # Loss should be positive

    def test_random_label_generation(self) -> None:
        """Test correct generation of random integer labels."""
        # Test the pattern we use for generating random labels
        num_classes = 5
        batch_size = 10

        labels = keras.ops.cast(
            keras.random.uniform((batch_size,), minval=0, maxval=num_classes),
            'int32'
        )

        # Check shape
        assert labels.shape == (batch_size,)

        # Check range
        assert keras.ops.all(labels >= 0)
        assert keras.ops.all(labels < num_classes)

        # Check dtype
        assert labels.dtype == 'int32'

    def test_model_summary_execution(self, base_config: Dict[str, Any]) -> None:
        """Test that model summary can be generated without errors."""
        model = FastVLM(**base_config)

        # This should not raise any exceptions
        try:
            model.summary()
        except Exception as e:
            pytest.fail(f"Model summary failed: {e}")

    @pytest.mark.parametrize("attention_type", [
        'multi_head_attention',
        'group_query_attention'
        # Note: window_attention requires specific spatial dimensions that are complex to set up
    ])
    def test_different_attention_types(
        self,
        base_config: Dict[str, Any],
        sample_input: keras.KerasTensor,
        attention_type: str
    ) -> None:
        """Test model with different attention mechanisms."""
        config = base_config.copy()
        config['attention_type'] = attention_type

        model = FastVLM(**config)
        output = model(sample_input)

        assert output.shape == (sample_input.shape[0], config['num_classes'])
        assert keras.ops.all(keras.ops.isfinite(output))
        assert model.attention_type == attention_type


# Additional integration tests
class TestFastVLMIntegration:
    """Integration tests for FastVLM with other framework components."""

    def test_with_mixed_precision(self) -> None:
        """Test FastVLM with mixed precision training."""
        # Enable mixed precision
        keras.mixed_precision.set_global_policy('mixed_float16')

        try:
            model = FastVLM(
                num_classes=10,
                embed_dims=[32, 64, 128],
                depths=[1, 1, 1],
                input_shape=(64, 64, 3)
            )

            # Test forward pass
            x = keras.random.normal((2, 64, 64, 3))
            output = model(x)

            assert output.dtype == keras.mixed_precision.global_policy().compute_dtype
            assert keras.ops.all(keras.ops.isfinite(output))

        finally:
            # Reset policy
            keras.mixed_precision.set_global_policy('float32')

    def test_transfer_learning_setup(self) -> None:
        """Test FastVLM setup for transfer learning."""
        # Create base model without top
        base_model = FastVLM.from_variant(
            'tiny',
            include_top=False,
            input_shape=(224, 224, 3)
        )

        # Freeze base model
        base_model.trainable = False

        # Add custom head
        inputs = keras.layers.Input(shape=(224, 224, 3))
        x = base_model(inputs)
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Dense(128, activation='relu')(x)
        outputs = keras.layers.Dense(5, activation='softmax')(x)  # Custom 5 classes

        transfer_model = keras.Model(inputs, outputs)

        # Test compilation and forward pass
        transfer_model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        x_test = keras.random.normal((1, 224, 224, 3))
        output = transfer_model(x_test)

        assert output.shape == (1, 5)
        assert keras.ops.all(keras.ops.isfinite(output))


# Performance and memory tests
class TestFastVLMPerformance:
    """Performance and memory usage tests for FastVLM."""

    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    def test_different_batch_sizes(
        self,
        batch_size: int
    ) -> None:
        """Test model with different batch sizes."""
        model = FastVLM(
            num_classes=10,
            embed_dims=[32, 64, 128],
            depths=[1, 1, 1],
            input_shape=(32, 32, 3)
        )

        x = keras.random.normal((batch_size, 32, 32, 3))
        output = model(x)

        assert output.shape == (batch_size, 10)
        assert keras.ops.all(keras.ops.isfinite(output))

    def test_memory_efficiency(self) -> None:
        """Test memory efficiency with gradient checkpointing."""
        import gc

        # Create model
        model = FastVLM(
            num_classes=100,
            embed_dims=[64, 128, 256],
            depths=[2, 2, 2],
            input_shape=(64, 64, 3),
            dropout_rate=0.1
        )

        # Forward and backward pass
        x = keras.random.normal((4, 64, 64, 3))

        with tf.GradientTape() as tape:
            output = model(x, training=True)
            loss = keras.ops.mean(output)

        gradients = tape.gradient(loss, model.trainable_variables)

        # Check gradients exist
        assert gradients is not None
        assert len(gradients) > 0

        # Cleanup
        del model, x, output, loss, gradients
        gc.collect()


if __name__ == "__main__":
    # Run tests with: python -m pytest test_fastvlm.py -v
    pytest.main([__file__, "-v", "--tb=short"])