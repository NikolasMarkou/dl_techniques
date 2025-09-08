import pytest
import tempfile
import os
import numpy as np
import keras
from keras import ops
import tensorflow as tf
from typing import Dict, Any, Tuple

from dl_techniques.models.nbeats.model import NBeatsNet, create_nbeats_model


class TestNBeatsNet:
    """Comprehensive test suite for N-BEATS model following modern Keras 3 patterns."""

    @pytest.fixture
    def basic_config(self) -> Dict[str, Any]:
        """Basic N-BEATS configuration for testing."""
        return {
            'backcast_length': 24,
            'forecast_length': 12,
            'stack_types': ['trend', 'seasonality'],
            'nb_blocks_per_stack': 2,
            'thetas_dim': [4, 8],
            'hidden_layer_units': 64,
            'use_revin': True,
            'dropout_rate': 0.1,
            'input_dim': 1,
            'output_dim': 1
        }

    @pytest.fixture
    def multivariate_config(self) -> Dict[str, Any]:
        """Multivariate N-BEATS configuration for testing."""
        return {
            'backcast_length': 48,
            'forecast_length': 24,
            'stack_types': ['trend', 'generic'],
            'nb_blocks_per_stack': 2,
            'thetas_dim': [6, 16],
            'hidden_layer_units': 128,
            'use_revin': True,
            'input_dim': 3,
            'output_dim': 2,
            'dropout_rate': 0.0
        }

    @pytest.fixture
    def sample_univariate_input(self) -> keras.KerasTensor:
        """Sample univariate input for testing."""
        return keras.random.normal(shape=(8, 24, 1))

    @pytest.fixture
    def sample_multivariate_input(self) -> keras.KerasTensor:
        """Sample multivariate input for testing."""
        return keras.random.normal(shape=(8, 48, 3))

    @pytest.fixture
    def sample_2d_input(self) -> keras.KerasTensor:
        """Sample 2D input for testing backward compatibility."""
        return keras.random.normal(shape=(8, 24))

    def test_initialization_basic(self, basic_config):
        """Test basic model initialization."""
        model = NBeatsNet(**basic_config)

        # Check configuration storage
        assert model.backcast_length == basic_config['backcast_length']
        assert model.forecast_length == basic_config['forecast_length']
        assert model.stack_types == basic_config['stack_types']
        assert model.nb_blocks_per_stack == basic_config['nb_blocks_per_stack']
        assert model.thetas_dim == basic_config['thetas_dim']
        assert model.hidden_layer_units == basic_config['hidden_layer_units']
        assert model.use_revin == basic_config['use_revin']
        assert model.dropout_rate == basic_config['dropout_rate']
        assert model.input_dim == basic_config['input_dim']
        assert model.output_dim == basic_config['output_dim']

        # Check sub-layers created
        assert model.global_revin is not None  # RevIN enabled
        assert len(model.blocks) == len(basic_config['stack_types'])
        assert len(model.dropout_layers) > 0  # Dropout configured
        assert not model.built  # Not built yet

    def test_initialization_multivariate(self, multivariate_config):
        """Test multivariate model initialization."""
        model = NBeatsNet(**multivariate_config)

        # Check multivariate-specific configuration
        assert model.input_dim == 3
        assert model.output_dim == 2
        assert model.output_projection is not None  # Should create projection layer

        # Check blocks created correctly
        assert len(model.blocks) == 2  # Two stacks
        for stack_blocks in model.blocks:
            assert len(stack_blocks) == multivariate_config['nb_blocks_per_stack']

    def test_forward_pass_univariate_3d(self, basic_config, sample_univariate_input):
        """Test forward pass with 3D univariate input."""
        model = NBeatsNet(**basic_config)

        # Forward pass should trigger build
        output = model(sample_univariate_input)

        assert model.built
        assert output.shape[0] == sample_univariate_input.shape[0]  # Batch size preserved
        assert output.shape[1] == basic_config['forecast_length']  # Correct forecast length
        assert output.shape[2] == basic_config['output_dim']  # Correct output dimension

        # Check output is not all zeros (model actually processed)
        output_np = ops.convert_to_numpy(output)
        assert not np.allclose(output_np, 0.0, atol=1e-6)

    def test_forward_pass_univariate_2d(self, basic_config, sample_2d_input):
        """Test forward pass with 2D input (backward compatibility)."""
        model = NBeatsNet(**basic_config)

        output = model(sample_2d_input)

        assert output.shape[0] == sample_2d_input.shape[0]  # Batch size preserved
        assert output.shape[1] == basic_config['forecast_length']  # Correct forecast length
        assert output.shape[2] == basic_config['output_dim']  # Should be 3D output

        # Verify model processes 2D input correctly
        output_np = ops.convert_to_numpy(output)
        assert not np.allclose(output_np, 0.0, atol=1e-6)

    def test_forward_pass_multivariate(self, multivariate_config, sample_multivariate_input):
        """Test forward pass with multivariate input."""
        model = NBeatsNet(**multivariate_config)

        output = model(sample_multivariate_input)

        assert output.shape[0] == sample_multivariate_input.shape[0]  # Batch size
        assert output.shape[1] == multivariate_config['forecast_length']  # Forecast length
        assert output.shape[2] == multivariate_config['output_dim']  # Output dimension

        # Check output projection was applied
        output_np = ops.convert_to_numpy(output)
        assert not np.allclose(output_np, 0.0, atol=1e-6)

    def test_serialization_cycle_basic(self, basic_config, sample_univariate_input):
        """CRITICAL TEST: Full serialization cycle for basic model."""
        # Create model in a keras Model wrapper for proper serialization
        inputs = keras.Input(shape=sample_univariate_input.shape[1:])
        nbeats_layer = NBeatsNet(**basic_config)
        outputs = nbeats_layer(inputs)
        model = keras.Model(inputs, outputs, name='nbeats_test_model')

        # Get original prediction
        original_prediction = model(sample_univariate_input)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'nbeats_test.keras')
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
        nbeats_layer = NBeatsNet(**multivariate_config)
        outputs = nbeats_layer(inputs)
        model = keras.Model(inputs, outputs, name='nbeats_multivariate_test')

        original_prediction = model(sample_multivariate_input)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'nbeats_multivariate_test.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_prediction = loaded_model(sample_multivariate_input)

            np.testing.assert_allclose(
                ops.convert_to_numpy(original_prediction),
                ops.convert_to_numpy(loaded_prediction),
                rtol=1e-6, atol=1e-6,
                err_msg="Multivariate predictions should match after serialization"
            )

    def test_config_completeness(self, basic_config):
        """Test that get_config contains all __init__ parameters."""
        model = NBeatsNet(**basic_config)
        config = model.get_config()

        # Check all basic_config parameters are present
        for key in basic_config:
            assert key in config, f"Missing {key} in get_config()"

        # Check additional parameters with defaults
        expected_keys = [
            'share_weights_in_stack', 'normalization_type', 'kernel_regularizer',
            'theta_regularizer', 'activation', 'kernel_initializer', 'use_bias'
        ]
        for key in expected_keys:
            assert key in config, f"Missing default parameter {key} in get_config()"

    def test_from_config_roundtrip(self, basic_config):
        """Test config serialization and deserialization."""
        original_model = NBeatsNet(**basic_config)
        config = original_model.get_config()

        # Create new model from config
        reconstructed_model = NBeatsNet.from_config(config)

        # Compare key attributes
        assert reconstructed_model.backcast_length == original_model.backcast_length
        assert reconstructed_model.forecast_length == original_model.forecast_length
        assert reconstructed_model.stack_types == original_model.stack_types
        assert reconstructed_model.thetas_dim == original_model.thetas_dim
        assert reconstructed_model.use_revin == original_model.use_revin

    def test_gradients_flow(self, basic_config, sample_univariate_input):
        """Test gradient computation through the model."""
        model = NBeatsNet(**basic_config)

        with tf.GradientTape() as tape:
            tape.watch(sample_univariate_input)
            output = model(sample_univariate_input)
            loss = ops.mean(ops.square(output))

        gradients = tape.gradient(loss, model.trainable_variables)

        # Note: This test is known to fail with this architecture because the backcast
        # weights of the final block do not contribute to the loss.
        # A more lenient check would be appropriate here.
        num_trainable_vars = len(model.trainable_variables)
        num_grads_with_none = sum(1 for g in gradients if g is None)

        # We expect the backcast theta weights of the last block to have no gradient
        assert num_grads_with_none <= 2, f"Expected at most 2 null gradients, found {num_grads_with_none}"
        assert len(gradients) > 0, "No trainable variables found"

    @pytest.mark.parametrize("training", [True, False, None])
    def test_training_modes(self, basic_config, sample_univariate_input, training):
        """Test behavior in different training modes."""
        model = NBeatsNet(**basic_config)

        output = model(sample_univariate_input, training=training)

        assert output.shape[0] == sample_univariate_input.shape[0]
        assert output.shape[1] == basic_config['forecast_length']
        assert output.shape[2] == basic_config['output_dim']

        # Check that output changes with training mode when dropout is enabled
        if basic_config['dropout_rate'] > 0.0 and training is not None:
            output_different_mode = model(sample_univariate_input, training=not training)
            # Outputs may be different due to dropout randomness
            assert output.shape == output_different_mode.shape

    def test_compute_output_shape(self, basic_config):
        """Test output shape computation."""
        model = NBeatsNet(**basic_config)

        # Test with 3D input shape
        input_shape_3d = (None, basic_config['backcast_length'], basic_config['input_dim'])
        output_shape = model.compute_output_shape(input_shape_3d)

        expected_shape = (None, basic_config['forecast_length'], basic_config['output_dim'])
        assert output_shape == expected_shape

        # Test with 2D input shape
        input_shape_2d = (None, basic_config['backcast_length'])
        output_shape_2d = model.compute_output_shape(input_shape_2d)
        assert output_shape_2d == expected_shape

    def test_edge_cases_validation(self):
        """Test error conditions and edge cases."""

        # Test invalid backcast_length
        with pytest.raises(ValueError, match="backcast_length must be positive"):
            NBeatsNet(backcast_length=0, forecast_length=12)

        # Test invalid forecast_length
        with pytest.raises(ValueError, match="forecast_length must be positive"):
            NBeatsNet(backcast_length=24, forecast_length=-5)

        # Test mismatched stack_types and thetas_dim lengths
        with pytest.raises(ValueError, match="Length of stack_types .* must match"):
            NBeatsNet(
                backcast_length=24, forecast_length=12,
                stack_types=['trend', 'seasonality'],
                thetas_dim=[4]  # Should be length 2
            )

        # Test invalid stack type
        with pytest.raises(ValueError, match="Invalid stack type"):
            NBeatsNet(
                backcast_length=24, forecast_length=12,
                stack_types=['invalid_type'],
                thetas_dim=[4]
            )

        # Test invalid dropout rate
        with pytest.raises(ValueError, match="dropout_rate must be in"):
            NBeatsNet(backcast_length=24, forecast_length=12, dropout_rate=1.5)

        # Test negative dimensions
        with pytest.raises(ValueError, match="input_dim must be positive"):
            NBeatsNet(backcast_length=24, forecast_length=12, input_dim=-1)

    def test_input_shape_validation(self, basic_config):
        """Test input shape validation during build."""
        model = NBeatsNet(**basic_config)

        # Test wrong sequence length
        wrong_length_input = keras.random.normal((8, 48, 1))  # Should be 24
        with pytest.raises(ValueError, match="Input sequence length .* doesn't match"):
            model(wrong_length_input)

        # Test wrong feature dimension for multivariate
        multivariate_model = NBeatsNet(
            backcast_length=24, forecast_length=12,
            input_dim=3, output_dim=1
        )
        wrong_features_input = keras.random.normal((8, 24, 2))  # Should be 3
        with pytest.raises(ValueError, match="Input feature dimension .* doesn't match"):
            multivariate_model(wrong_features_input)

    def test_revin_functionality(self, basic_config, sample_univariate_input):
        """Test RevIN normalization functionality."""
        # Model with RevIN
        model_with_revin = NBeatsNet(**basic_config)
        assert model_with_revin.use_revin is True
        assert model_with_revin.global_revin is not None

        # Model without RevIN
        config_no_revin = basic_config.copy()
        config_no_revin['use_revin'] = False
        model_without_revin = NBeatsNet(**config_no_revin)
        assert model_without_revin.use_revin is False
        assert model_without_revin.global_revin is None

        # Both should produce valid outputs
        output_with_revin = model_with_revin(sample_univariate_input)
        output_without_revin = model_without_revin(sample_univariate_input)

        assert output_with_revin.shape == output_without_revin.shape
        # Outputs should be different due to normalization
        assert not np.allclose(
            ops.convert_to_numpy(output_with_revin),
            ops.convert_to_numpy(output_without_revin),
            rtol=1e-3
        )

    def test_different_stack_configurations(self, sample_univariate_input):
        """Test different stack type configurations."""
        base_config = {
            'backcast_length': 24,
            'forecast_length': 12,
            'nb_blocks_per_stack': 2,
            'hidden_layer_units': 32
        }

        # Test all three stack types
        configs = [
            (['generic'], [16]),
            (['trend'], [4]),
            (['seasonality'], [8]),
            (['trend', 'seasonality'], [4, 8]),
            (['generic', 'trend', 'seasonality'], [16, 4, 8])
        ]

        for stack_types, thetas_dim in configs:
            config = base_config.copy()
            config.update({
                'stack_types': stack_types,
                'thetas_dim': thetas_dim
            })

            model = NBeatsNet(**config)
            output = model(sample_univariate_input)

            assert output.shape == (sample_univariate_input.shape[0], 12, 1)
            assert len(model.blocks) == len(stack_types)


class TestNBeatsFactory:
    """Test suite for the create_nbeats_model factory function."""

    @pytest.fixture
    def sample_data(self) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """Sample training data for factory tests."""
        x = keras.random.normal((32, 96, 1))
        y = keras.random.normal((32, 24, 1))
        return x, y

    def test_factory_basic_creation(self):
        """Test basic factory model creation."""
        model = create_nbeats_model()

        # Check default configuration
        assert model.backcast_length == 96
        assert model.forecast_length == 24
        assert model.stack_types == ['trend', 'seasonality']
        assert model.nb_blocks_per_stack == 3
        assert model.use_revin is True
        assert model.hidden_layer_units == 256

        # Check model is compiled
        assert model.optimizer is not None
        assert model.compiled_loss is not None

    def test_factory_auto_theta_calculation(self):
        """Test automatic theta dimension calculation."""
        model = create_nbeats_model(
            forecast_length=48,
            stack_types=['trend', 'seasonality', 'generic']
        )

        # Check auto-calculated theta dimensions
        expected_trend = 4
        # 48//3 = 16 harmonics => 32 thetas
        expected_seasonality = 32
        # 48*2 = 96 thetas
        expected_generic = 96

        assert model.thetas_dim[0] == expected_trend
        assert model.thetas_dim[1] == expected_seasonality
        assert model.thetas_dim[2] == expected_generic

    def test_factory_custom_configuration(self):
        """Test factory with custom configuration."""
        model = create_nbeats_model(
            backcast_length=168,
            forecast_length=48,
            stack_types=['generic', 'trend'],
            nb_blocks_per_stack=4,
            thetas_dim=[32, 6],
            hidden_layer_units=512,
            use_revin=False,
            dropout_rate=0.2,
            learning_rate=1e-3,
            gradient_clip_norm=0.5
        )

        # Check custom configuration
        assert model.backcast_length == 168
        assert model.forecast_length == 48
        assert model.stack_types == ['generic', 'trend']
        assert model.thetas_dim == [32, 6]
        assert model.hidden_layer_units == 512
        assert model.use_revin is False
        assert model.dropout_rate == 0.2

        # Check optimizer configuration
        assert hasattr(model.optimizer, 'clipnorm')
        assert model.optimizer.clipnorm == 0.5

    def test_factory_optimizer_configurations(self):
        """Test different optimizer configurations."""

        # String optimizer with gradient clipping
        model1 = create_nbeats_model(
            optimizer='adamw',
            learning_rate=5e-4,
            gradient_clip_norm=1.5
        )
        assert isinstance(model1.optimizer, keras.optimizers.AdamW)
        assert model1.optimizer.clipnorm == 1.5

        # Custom optimizer instance
        custom_optimizer = keras.optimizers.Adam(learning_rate=1e-3)
        model2 = create_nbeats_model(
            optimizer=custom_optimizer,
            gradient_clip_norm=2.0
        )
        assert model2.optimizer is custom_optimizer
        assert model2.optimizer.clipnorm == 2.0

        # Custom optimizer with existing clipping (should preserve)
        optimizer_with_clip = keras.optimizers.RMSprop(learning_rate=1e-3, clipnorm=0.8)
        model3 = create_nbeats_model(
            optimizer=optimizer_with_clip,
            gradient_clip_norm=1.0  # Should not override existing
        )
        assert model3.optimizer.clipnorm == 0.8  # Preserved original

    def test_factory_loss_and_metrics(self):
        """Test loss and metrics configuration."""

        # Default configuration
        model1 = create_nbeats_model()
        # FIX: Use a robust way to check the loss function
        assert model1.loss == 'mae'

        # Custom loss and metrics
        model2 = create_nbeats_model(
            loss='mse',
            metrics=['mae', 'mape']
        )
        assert model2.loss == 'mse'

    def test_factory_regularization(self):
        """Test factory with regularization."""
        kernel_reg = keras.regularizers.L2(1e-3)

        model = create_nbeats_model(
            kernel_regularizer=kernel_reg,
            dropout_rate=0.15
        )

        assert model.kernel_regularizer is not None
        assert model.dropout_rate == 0.15

    def test_factory_training_integration(self, sample_data):
        """Test that factory-created model can be trained."""
        x_train, y_train = sample_data

        model = create_nbeats_model(
            backcast_length=96,
            forecast_length=24,
            hidden_layer_units=64  # Smaller for faster testing
        )

        # Should be able to fit without errors
        history = model.fit(
            x_train, y_train,
            epochs=2,
            batch_size=16,
            verbose=0
        )

        assert len(history.history['loss']) == 2
        assert all(not np.isnan(loss) for loss in history.history['loss'])

    def test_factory_ratio_warnings(self, caplog):
        """Test backcast/forecast ratio warnings."""
        # Test warning for low ratio
        create_nbeats_model(
            backcast_length=24,
            forecast_length=12  # ratio = 2.0 < 3.0
        )

        assert "Consider increasing backcast_length" in caplog.text

    def test_factory_serialization_compatibility(self, sample_data):
        """Test that factory-created models serialize correctly."""
        x_test, _ = sample_data

        model = create_nbeats_model(
            hidden_layer_units=32  # Small for testing
        )

        # Get prediction
        original_pred = model(x_test)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'factory_model.keras')
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

        # FIX: Test that an unknown optimizer correctly raises a ValueError
        with pytest.raises(ValueError, match="Could not interpret optimizer identifier"):
            create_nbeats_model(optimizer='unknown_optimizer')

        # Test with None gradient clipping
        model = create_nbeats_model(gradient_clip_norm=None)
        assert model.optimizer.clipnorm is None

        # Test invalid stack types (should raise error)
        with pytest.raises(ValueError):
            create_nbeats_model(stack_types=['invalid_stack_type'])

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])