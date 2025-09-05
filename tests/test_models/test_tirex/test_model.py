import pytest
import tempfile
import os
import numpy as np
import keras
from keras import ops
import tensorflow as tf
from typing import Dict, Any, Tuple, List

from dl_techniques.models.tirex.model import TiRexCore, create_tirex_model, create_tirex_by_variant


class TestTiRexCore:
    """Comprehensive test suite for TiRex model following modern Keras 3 patterns."""

    @pytest.fixture
    def basic_config(self) -> Dict[str, Any]:
        """Basic TiRex configuration for testing."""
        return {
            'patch_size': 8,
            'embed_dim': 64,
            'num_blocks': 3,
            'num_heads': 4,
            'lstm_units': 64,
            'ff_dim': 256,
            'block_types': ['mixed', 'transformer', 'lstm'],
            'quantile_levels': [0.1, 0.5, 0.9],
            'prediction_length': 12,
            'dropout_rate': 0.1,
            'use_layer_norm': True
        }

    @pytest.fixture
    def multivariate_config(self) -> Dict[str, Any]:
        """Multivariate TiRex configuration for testing."""
        return {
            'patch_size': 16,
            'embed_dim': 128,
            'num_blocks': 4,
            'num_heads': 8,
            'lstm_units': 128,
            'ff_dim': 512,
            'block_types': ['mixed', 'mixed', 'transformer', 'lstm'],
            'quantile_levels': [0.1, 0.2, 0.5, 0.8, 0.9],
            'prediction_length': 24,
            'dropout_rate': 0.0,
            'use_layer_norm': True
        }

    @pytest.fixture
    def sample_univariate_input(self) -> keras.KerasTensor:
        """Sample univariate input for testing."""
        return keras.random.normal(shape=(8, 96, 1))

    @pytest.fixture
    def sample_multivariate_input(self) -> keras.KerasTensor:
        """Sample multivariate input for testing."""
        return keras.random.normal(shape=(8, 128, 3))

    @pytest.fixture
    def sample_2d_input(self) -> keras.KerasTensor:
        """Sample 2D input for testing backward compatibility."""
        return keras.random.normal(shape=(8, 96))

    def test_initialization_basic(self, basic_config):
        """Test basic model initialization."""
        model = TiRexCore(**basic_config)

        # Check configuration storage
        assert model.patch_size == basic_config['patch_size']
        assert model.embed_dim == basic_config['embed_dim']
        assert model.num_blocks == basic_config['num_blocks']
        assert model.num_heads == basic_config['num_heads']
        assert model.lstm_units == basic_config['lstm_units']
        assert model.ff_dim == basic_config['ff_dim']
        assert model.block_types == basic_config['block_types']
        assert model.quantile_levels == basic_config['quantile_levels']
        assert model.prediction_length == basic_config['prediction_length']
        assert model.dropout_rate == basic_config['dropout_rate']
        assert model.use_layer_norm == basic_config['use_layer_norm']

        # Check sub-layers created
        assert model.scaler is not None
        assert model.patch_embedding is not None
        assert model.input_projection is not None
        assert len(model.blocks) == basic_config['num_blocks']
        assert model.output_norm is not None
        assert model.quantile_head is not None

    def test_initialization_multivariate(self, multivariate_config):
        """Test multivariate model initialization."""
        model = TiRexCore(**multivariate_config)

        # Check multivariate-specific configuration
        assert model.embed_dim == multivariate_config['embed_dim']
        assert len(model.quantile_levels) == 5  # 5 quantiles
        assert model.prediction_length == 24

        # Check blocks created correctly
        assert len(model.blocks) == 4  # Four blocks
        for i, block_type in enumerate(multivariate_config['block_types']):
            # Each block should have correct type stored
            assert model.block_types[i] == block_type

    def test_forward_pass_univariate_3d(self, basic_config, sample_univariate_input):
        """Test forward pass with 3D univariate input."""
        model = TiRexCore(**basic_config)

        # Forward pass
        output = model(sample_univariate_input)

        # Check output shape: [batch_size, num_quantiles, prediction_length]
        assert output.shape[0] == sample_univariate_input.shape[0]  # Batch size preserved
        assert output.shape[1] == len(basic_config['quantile_levels'])  # Number of quantiles
        assert output.shape[2] == basic_config['prediction_length']  # Prediction length

        # Check output is not all zeros (model actually processed)
        output_np = ops.convert_to_numpy(output)
        assert not np.allclose(output_np, 0.0, atol=1e-6)

    def test_forward_pass_univariate_2d(self, basic_config, sample_2d_input):
        """Test forward pass with 2D input (backward compatibility)."""
        model = TiRexCore(**basic_config)

        output = model(sample_2d_input)

        # Should expand 2D to 3D and process correctly
        assert output.shape[0] == sample_2d_input.shape[0]  # Batch size preserved
        assert output.shape[1] == len(basic_config['quantile_levels'])  # Number of quantiles
        assert output.shape[2] == basic_config['prediction_length']  # Prediction length

        # Verify model processes 2D input correctly
        output_np = ops.convert_to_numpy(output)
        assert not np.allclose(output_np, 0.0, atol=1e-6)

    def test_forward_pass_multivariate(self, multivariate_config, sample_multivariate_input):
        """Test forward pass with multivariate input."""
        model = TiRexCore(**multivariate_config)

        output = model(sample_multivariate_input)

        assert output.shape[0] == sample_multivariate_input.shape[0]  # Batch size
        assert output.shape[1] == len(multivariate_config['quantile_levels'])  # Num quantiles
        assert output.shape[2] == multivariate_config['prediction_length']  # Prediction length

        # Check output is meaningful
        output_np = ops.convert_to_numpy(output)
        assert not np.allclose(output_np, 0.0, atol=1e-6)

    def test_serialization_cycle_basic(self, basic_config, sample_univariate_input):
        """CRITICAL TEST: Full serialization cycle for basic model."""
        # Create model in a keras Model wrapper for proper serialization
        inputs = keras.Input(shape=sample_univariate_input.shape[1:])
        tirex_layer = TiRexCore(**basic_config)
        outputs = tirex_layer(inputs)
        model = keras.Model(inputs, outputs, name='tirex_test_model')

        # Get original prediction
        original_prediction = model(sample_univariate_input)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'tirex_test.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_prediction = loaded_model(sample_univariate_input)

            # Verify identical outputs
            np.testing.assert_allclose(
                ops.convert_to_numpy(original_prediction),
                ops.convert_to_numpy(loaded_prediction),
                rtol=1e-6, atol=1e-6,
                err_msg="Predictions should match after serialization"
            )

    def test_serialization_cycle_multivariate(self, multivariate_config, sample_multivariate_input):
        """CRITICAL TEST: Full serialization cycle for multivariate model."""
        inputs = keras.Input(shape=sample_multivariate_input.shape[1:])
        tirex_layer = TiRexCore(**multivariate_config)
        outputs = tirex_layer(inputs)
        model = keras.Model(inputs, outputs, name='tirex_multivariate_test')

        original_prediction = model(sample_multivariate_input)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'tirex_multivariate_test.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_prediction = loaded_model(sample_multivariate_input)

            np.testing.assert_allclose(
                ops.convert_to_numpy(original_prediction),
                ops.convert_to_numpy(loaded_prediction),
                rtol=1e-6, atol=1e-6,
                err_msg="Multivariate predictions should match after serialization"
            )

    def test_from_config_roundtrip(self, basic_config):
        """Test config serialization and deserialization."""
        original_model = TiRexCore(**basic_config)
        config = original_model.get_config()

        # Create new model from config
        reconstructed_model = TiRexCore.from_config(config)

        # Compare key attributes
        assert reconstructed_model.patch_size == original_model.patch_size
        assert reconstructed_model.embed_dim == original_model.embed_dim
        assert reconstructed_model.num_blocks == original_model.num_blocks
        assert reconstructed_model.quantile_levels == original_model.quantile_levels
        assert reconstructed_model.prediction_length == original_model.prediction_length
        assert reconstructed_model.block_types == original_model.block_types

    def test_gradients_flow(self, basic_config, sample_univariate_input):
        """Test gradient computation through the model."""
        model = TiRexCore(**basic_config)

        with tf.GradientTape() as tape:
            tape.watch(sample_univariate_input)
            output = model(sample_univariate_input)
            loss = ops.mean(ops.square(output))

        gradients = tape.gradient(loss, model.trainable_variables)

        # Check that most gradients are not None
        num_trainable_vars = len(model.trainable_variables)
        num_grads_with_none = sum(1 for g in gradients if g is None)

        # Allow some gradients to be None (e.g., from unused paths)
        assert num_grads_with_none < num_trainable_vars // 2, f"Too many null gradients: {num_grads_with_none}/{num_trainable_vars}"
        assert len(gradients) > 0, "No trainable variables found"

    @pytest.mark.parametrize("training", [True, False, None])
    def test_training_modes(self, basic_config, sample_univariate_input, training):
        """Test behavior in different training modes."""
        model = TiRexCore(**basic_config)

        output = model(sample_univariate_input, training=training)

        assert output.shape[0] == sample_univariate_input.shape[0]
        assert output.shape[1] == len(basic_config['quantile_levels'])
        assert output.shape[2] == basic_config['prediction_length']

        # Check that output changes with training mode when dropout is enabled
        if basic_config['dropout_rate'] > 0.0 and training is not None:
            output_different_mode = model(sample_univariate_input, training=not training)
            # Outputs may be different due to dropout randomness
            assert output.shape == output_different_mode.shape

    def test_edge_cases_validation(self):
        """Test error conditions and edge cases."""

        # Test invalid patch_size
        with pytest.raises(ValueError, match="patch_size must be positive"):
            TiRexCore(patch_size=0, embed_dim=64, num_blocks=2, prediction_length=12)

        # Test invalid embed_dim
        with pytest.raises(ValueError, match="embed_dim must be positive"):
            TiRexCore(patch_size=8, embed_dim=-64, num_blocks=2, prediction_length=12)

        # Test invalid num_blocks
        with pytest.raises(ValueError, match="num_blocks must be positive"):
            TiRexCore(patch_size=8, embed_dim=64, num_blocks=0, prediction_length=12)

        # Test invalid prediction_length
        with pytest.raises(ValueError, match="prediction_length must be positive"):
            TiRexCore(patch_size=8, embed_dim=64, num_blocks=2, prediction_length=-5)

        # Test mismatched block_types and num_blocks
        with pytest.raises(ValueError, match="Length of block_types .* must match"):
            TiRexCore(
                patch_size=8, embed_dim=64, num_blocks=3,
                block_types=['mixed', 'lstm'],  # Should be length 3
                prediction_length=12
            )

    def test_input_shape_validation(self, basic_config):
        """Test input shape validation during forward pass."""
        model = TiRexCore(**basic_config)

        # Test invalid input shape (1D)
        invalid_1d_input = keras.random.normal((8, 96))  # Missing feature dimension
        # This should work as it gets expanded to 3D

        # Test invalid input shape (4D)
        invalid_4d_input = keras.random.normal((8, 96, 1, 1))
        with pytest.raises(ValueError, match="Input must be 2D or 3D tensor"):
            model(invalid_4d_input)

    def test_different_block_configurations(self, sample_univariate_input):
        """Test different block type configurations."""
        base_config = {
            'patch_size': 8,
            'embed_dim': 64,
            'num_blocks': 2,
            'num_heads': 4,
            'prediction_length': 12,
            'quantile_levels': [0.1, 0.5, 0.9]
        }

        # Test all three block types
        configs = [
            ['lstm', 'lstm'],
            ['transformer', 'transformer'],
            ['mixed', 'mixed'],
            ['lstm', 'transformer'],
            ['transformer', 'mixed'],
            ['mixed', 'lstm']
        ]

        for block_types in configs:
            config = base_config.copy()
            config['block_types'] = block_types

            model = TiRexCore(**config)
            output = model(sample_univariate_input)

            expected_shape = (sample_univariate_input.shape[0], 3, 12)  # batch, quantiles, pred_length
            assert output.shape == expected_shape
            assert len(model.blocks) == len(block_types)

    def test_predict_quantiles_method(self, basic_config, sample_univariate_input):
        """Test the predict_quantiles method."""
        model = TiRexCore(**basic_config)

        # Convert to numpy for predict_quantiles
        input_np = ops.convert_to_numpy(sample_univariate_input)

        # Test with default quantiles
        quantile_preds, mean_preds = model.predict_quantiles(input_np)

        assert quantile_preds.shape[0] == input_np.shape[0]  # Batch size
        assert quantile_preds.shape[1] == len(basic_config['quantile_levels'])  # All quantiles
        assert quantile_preds.shape[2] == basic_config['prediction_length']  # Prediction length

        assert mean_preds.shape[0] == input_np.shape[0]  # Batch size
        assert mean_preds.shape[1] == basic_config['prediction_length']  # Prediction length

        # Test with custom quantiles
        custom_quantiles = [0.25, 0.75]
        quantile_preds_custom, _ = model.predict_quantiles(input_np, quantile_levels=custom_quantiles)
        assert quantile_preds_custom.shape[1] == 2  # Two quantiles

    def test_variant_creation(self):
        """Test creation from variants."""

        # Test all available variants
        variants = ['tiny', 'small', 'medium', 'large']

        for variant in variants:
            model = TiRexCore.from_variant(variant, prediction_length=24)

            # Check that model was created with variant config
            variant_config = TiRexCore.MODEL_VARIANTS[variant]
            assert model.patch_size == variant_config['patch_size']
            assert model.embed_dim == variant_config['embed_dim']
            assert model.num_blocks == variant_config['num_blocks']
            assert model.num_heads == variant_config['num_heads']
            assert model.dropout_rate == variant_config['dropout_rate']
            assert model.prediction_length == 24

    def test_invalid_variant(self):
        """Test error handling for invalid variant."""
        with pytest.raises(ValueError, match="Unknown variant"):
            TiRexCore.from_variant("invalid_variant")

    def test_layer_normalization_toggle(self, basic_config, sample_univariate_input):
        """Test layer normalization functionality."""
        # Model with layer norm
        config_with_norm = basic_config.copy()
        config_with_norm['use_layer_norm'] = True
        model_with_norm = TiRexCore(**config_with_norm)

        # Model without layer norm
        config_no_norm = basic_config.copy()
        config_no_norm['use_layer_norm'] = False
        model_without_norm = TiRexCore(**config_no_norm)

        # Both should produce valid outputs
        output_with_norm = model_with_norm(sample_univariate_input)
        output_without_norm = model_without_norm(sample_univariate_input)

        assert output_with_norm.shape == output_without_norm.shape
        # Outputs should be different due to normalization
        assert not np.allclose(
            ops.convert_to_numpy(output_with_norm),
            ops.convert_to_numpy(output_without_norm),
            rtol=1e-3
        )


class TestTiRexFactory:
    """Test suite for the TiRex factory functions."""

    @pytest.fixture
    def sample_data(self) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """Sample training data for factory tests."""
        x = keras.random.normal((32, 128, 1))
        y = keras.random.normal((32, 24, 1))
        return x, y

    def test_create_tirex_model_basic(self):
        """Test basic factory model creation."""
        model = create_tirex_model(
            input_length=96,
            prediction_length=24
        )

        # Check default configuration
        assert model.patch_size == 16
        assert model.embed_dim == 256
        assert model.num_blocks == 6
        assert model.num_heads == 8
        assert model.prediction_length == 24
        assert model.quantile_levels == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    def test_create_tirex_model_custom(self):
        """Test factory with custom configuration."""
        custom_quantiles = [0.1, 0.5, 0.9]
        block_types = ['lstm', 'transformer', 'mixed']

        model = create_tirex_model(
            input_length=168,
            prediction_length=48,
            patch_size=12,
            embed_dim=128,
            num_blocks=3,
            num_heads=8,
            quantile_levels=custom_quantiles,
            block_types=block_types,
            dropout_rate=0.2
        )

        # Check custom configuration
        assert model.prediction_length == 48
        assert model.patch_size == 12
        assert model.embed_dim == 128
        assert model.num_blocks == 3
        assert model.quantile_levels == custom_quantiles
        assert model.block_types == block_types
        assert model.dropout_rate == 0.2

    def test_create_tirex_by_variant_basic(self):
        """Test variant factory creation."""
        model = create_tirex_by_variant()

        # Check default variant (medium)
        medium_config = TiRexCore.MODEL_VARIANTS['medium']
        assert model.patch_size == medium_config['patch_size']
        assert model.embed_dim == medium_config['embed_dim']
        assert model.num_blocks == medium_config['num_blocks']
        assert model.dropout_rate == medium_config['dropout_rate']

    def test_create_tirex_by_variant_custom(self):
        """Test variant factory with custom parameters."""
        model = create_tirex_by_variant(
            variant='large',
            input_length=256,
            prediction_length=48,
            quantile_levels=[0.05, 0.25, 0.5, 0.75, 0.95]
        )

        # Check large variant config
        large_config = TiRexCore.MODEL_VARIANTS['large']
        assert model.patch_size == large_config['patch_size']
        assert model.embed_dim == large_config['embed_dim']
        assert model.num_blocks == large_config['num_blocks']
        assert model.prediction_length == 48
        assert len(model.quantile_levels) == 5

    def test_factory_all_variants(self):
        """Test factory creation for all variants."""
        variants = ['tiny', 'small', 'medium', 'large']

        for variant in variants:
            model = create_tirex_by_variant(
                variant=variant,
                input_length=96,
                prediction_length=24
            )

            variant_config = TiRexCore.MODEL_VARIANTS[variant]
            assert model.patch_size == variant_config['patch_size']
            assert model.embed_dim == variant_config['embed_dim']
            assert model.num_blocks == variant_config['num_blocks']

    def test_factory_model_building(self, sample_data):
        """Test that factory models are properly built."""
        x_train, _ = sample_data

        model = create_tirex_model(
            input_length=128,
            prediction_length=24,
            embed_dim=64  # Smaller for testing
        )

        # Model should handle input without error
        output = model(x_train)

        assert output.shape[0] == x_train.shape[0]  # Batch size
        assert output.shape[1] == 9  # Default quantiles
        assert output.shape[2] == 24  # Prediction length

    def test_factory_serialization_compatibility(self, sample_data):
        """Test that factory-created models serialize correctly."""
        x_test, _ = sample_data

        model = create_tirex_by_variant(
            variant='tiny',  # Small for testing
            input_length=128,
            prediction_length=12
        )

        # Get prediction
        original_pred = model(x_test)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'factory_tirex_model.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_pred = loaded_model(x_test)

            np.testing.assert_allclose(
                ops.convert_to_numpy(original_pred),
                ops.convert_to_numpy(loaded_pred),
                rtol=1e-6, atol=1e-6,
                err_msg="Factory model predictions should match after serialization"
            )

    def test_factory_edge_cases(self):
        """Test factory function edge cases."""

        # Test invalid variant
        with pytest.raises(ValueError, match="Unknown variant"):
            create_tirex_by_variant(variant='invalid_variant')

        # Test with minimal configuration
        model = create_tirex_model(
            input_length=24,
            prediction_length=6,
            patch_size=4,
            embed_dim=32,
            num_blocks=1
        )

        assert model.patch_size == 4
        assert model.embed_dim == 32
        assert model.num_blocks == 1

    def test_factory_quantile_configurations(self):
        """Test different quantile configurations."""

        # Single quantile (median only)
        model_single = create_tirex_model(
            input_length=96,
            prediction_length=24,
            quantile_levels=[0.5]
        )
        assert len(model_single.quantile_levels) == 1

        # Many quantiles
        many_quantiles = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
        model_many = create_tirex_model(
            input_length=96,
            prediction_length=24,
            quantile_levels=many_quantiles
        )
        assert len(model_many.quantile_levels) == 9

    def test_factory_block_type_configurations(self):
        """Test different block type configurations."""

        # All LSTM
        model_lstm = create_tirex_model(
            input_length=96,
            prediction_length=24,
            num_blocks=3,
            block_types=['lstm', 'lstm', 'lstm']
        )
        assert model_lstm.block_types == ['lstm', 'lstm', 'lstm']

        # All Transformer
        model_transformer = create_tirex_model(
            input_length=96,
            prediction_length=24,
            num_blocks=2,
            block_types=['transformer', 'transformer']
        )
        assert model_transformer.block_types == ['transformer', 'transformer']

        # Mixed configuration
        model_mixed = create_tirex_model(
            input_length=96,
            prediction_length=24,
            num_blocks=4,
            block_types=['lstm', 'mixed', 'transformer', 'mixed']
        )
        assert model_mixed.block_types == ['lstm', 'mixed', 'transformer', 'mixed']

    def test_factory_parameter_passing(self):
        """Test that factory functions properly pass through parameters."""

        model = create_tirex_by_variant(
            variant='small',
            input_length=192,
            prediction_length=36,
            lstm_units=256,  # Override default
            ff_dim=1024,  # Override default
            dropout_rate=0.15,  # Override default
            use_layer_norm=False  # Override default
        )

        # Check overridden parameters
        assert model.lstm_units == 256
        assert model.ff_dim == 1024
        assert model.dropout_rate == 0.15
        assert model.use_layer_norm is False

        # Check variant parameters still applied
        small_config = TiRexCore.MODEL_VARIANTS['small']
        assert model.patch_size == small_config['patch_size']
        assert model.embed_dim == small_config['embed_dim']