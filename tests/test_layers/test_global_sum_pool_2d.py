import pytest
import tempfile
import os
import numpy as np
import tensorflow as tf
import keras
from typing import Any, Dict


from dl_techniques.layers.global_sum_pool_2d import GlobalSumPooling2D


class TestGlobalSumPooling2D:
    """Comprehensive test suite for GlobalSumPooling2D layer."""

    @pytest.fixture
    def layer_config_channels_last(self) -> Dict[str, Any]:
        """Standard configuration for channels_last testing."""
        return {
            'keepdims': False,
            'data_format': 'channels_last'
        }

    @pytest.fixture
    def layer_config_channels_first(self) -> Dict[str, Any]:
        """Standard configuration for channels_first testing."""
        return {
            'keepdims': False,
            'data_format': 'channels_first'
        }

    @pytest.fixture
    def layer_config_keepdims(self) -> Dict[str, Any]:
        """Configuration with keepdims=True for testing."""
        return {
            'keepdims': True,
            'data_format': 'channels_last'
        }

    @pytest.fixture
    def sample_input_channels_last(self) -> keras.KerasTensor:
        """Sample input for channels_last format (batch, height, width, channels)."""
        return keras.random.normal(shape=(4, 8, 8, 32))

    @pytest.fixture
    def sample_input_channels_first(self) -> keras.KerasTensor:
        """Sample input for channels_first format (batch, channels, height, width)."""
        return keras.random.normal(shape=(4, 32, 8, 8))

    def test_initialization_default(self):
        """Test layer initialization with default parameters."""
        layer = GlobalSumPooling2D()

        assert hasattr(layer, 'keepdims')
        assert hasattr(layer, 'data_format')
        assert layer.keepdims == False
        assert layer.data_format == keras.backend.image_data_format()
        assert not layer.built

    def test_initialization_custom_params(self, layer_config_channels_last):
        """Test layer initialization with custom parameters."""
        layer = GlobalSumPooling2D(**layer_config_channels_last)

        assert layer.keepdims == layer_config_channels_last['keepdims']
        assert layer.data_format == layer_config_channels_last['data_format']
        assert not layer.built

    def test_initialization_invalid_data_format(self):
        """Test initialization with invalid data format."""
        with pytest.raises(ValueError, match="data_format must be"):
            GlobalSumPooling2D(data_format="invalid_format")

    def test_forward_pass_channels_last(self, layer_config_channels_last, sample_input_channels_last):
        """Test forward pass with channels_last format."""
        layer = GlobalSumPooling2D(**layer_config_channels_last)

        output = layer(sample_input_channels_last)

        assert layer.built
        assert output.shape[0] == sample_input_channels_last.shape[0]  # Batch size preserved
        assert output.shape[1] == sample_input_channels_last.shape[3]  # Channel dimension preserved
        assert len(output.shape) == 2  # Spatial dimensions removed (keepdims=False)

    def test_forward_pass_channels_first(self, layer_config_channels_first, sample_input_channels_first):
        """Test forward pass with channels_first format."""
        layer = GlobalSumPooling2D(**layer_config_channels_first)

        output = layer(sample_input_channels_first)

        assert layer.built
        assert output.shape[0] == sample_input_channels_first.shape[0]  # Batch size preserved
        assert output.shape[1] == sample_input_channels_first.shape[1]  # Channel dimension preserved
        assert len(output.shape) == 2  # Spatial dimensions removed (keepdims=False)

    def test_forward_pass_keepdims(self, layer_config_keepdims, sample_input_channels_last):
        """Test forward pass with keepdims=True."""
        layer = GlobalSumPooling2D(**layer_config_keepdims)

        output = layer(sample_input_channels_last)

        assert layer.built
        assert output.shape[0] == sample_input_channels_last.shape[0]  # Batch size preserved
        assert output.shape[1] == 1  # Height dimension kept as 1
        assert output.shape[2] == 1  # Width dimension kept as 1
        assert output.shape[3] == sample_input_channels_last.shape[3]  # Channel dimension preserved
        assert len(output.shape) == 4  # All dimensions kept

    def test_serialization_cycle_channels_last(self, layer_config_channels_last, sample_input_channels_last):
        """CRITICAL TEST: Full serialization cycle with channels_last."""
        # Create model with custom layer
        inputs = keras.Input(shape=sample_input_channels_last.shape[1:])
        outputs = GlobalSumPooling2D(**layer_config_channels_last)(inputs)
        model = keras.Model(inputs, outputs)

        # Get original prediction
        original_pred = model(sample_input_channels_last)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_model.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_pred = loaded_model(sample_input_channels_last)

            # Verify identical predictions
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_pred),
                keras.ops.convert_to_numpy(loaded_pred),
                rtol=1e-6, atol=1e-6,
                err_msg="Predictions differ after serialization"
            )

    def test_serialization_cycle_channels_first(self, layer_config_channels_first, sample_input_channels_first):
        """CRITICAL TEST: Full serialization cycle with channels_first."""
        # Create model with custom layer
        inputs = keras.Input(shape=sample_input_channels_first.shape[1:])
        outputs = GlobalSumPooling2D(**layer_config_channels_first)(inputs)
        model = keras.Model(inputs, outputs)

        # Get original prediction
        original_pred = model(sample_input_channels_first)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_model.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_pred = loaded_model(sample_input_channels_first)

            # Verify identical predictions
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_pred),
                keras.ops.convert_to_numpy(loaded_pred),
                rtol=1e-6, atol=1e-6,
                err_msg="Predictions differ after serialization"
            )

    def test_config_completeness(self, layer_config_channels_last):
        """Test that get_config contains all __init__ parameters."""
        layer = GlobalSumPooling2D(**layer_config_channels_last)
        config = layer.get_config()

        # Check all config parameters are present
        for key in layer_config_channels_last:
            assert key in config, f"Missing {key} in get_config()"

        # Verify config values
        assert config['keepdims'] == layer_config_channels_last['keepdims']
        assert config['data_format'] == layer_config_channels_last['data_format']

    def test_gradients_flow_through_layer(self, layer_config_channels_last, sample_input_channels_last):
        """Test that gradients flow through the layer."""
        # Note: GlobalSumPooling2D has no trainable parameters,
        # but we can test that gradients flow through it
        layer = GlobalSumPooling2D(**layer_config_channels_last)

        # Create a simple model with trainable weights after the pooling layer
        inputs = keras.Input(shape=sample_input_channels_last.shape[1:])
        x = layer(inputs)
        outputs = keras.layers.Dense(1)(x)
        model = keras.Model(inputs, outputs)

        # Use persistent tape to compute multiple gradients
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(sample_input_channels_last)
            output = model(sample_input_channels_last)
            loss = keras.ops.mean(keras.ops.square(output))

        # Test gradients flow to input
        input_gradients = tape.gradient(loss, sample_input_channels_last)
        assert input_gradients is not None

        # Test gradients flow to model weights
        weight_gradients = tape.gradient(loss, model.trainable_variables)
        assert all(g is not None for g in weight_gradients)
        assert len(weight_gradients) > 0

        # Clean up persistent tape
        del tape

    @pytest.mark.parametrize("training", [True, False, None])
    def test_training_modes(self, layer_config_channels_last, sample_input_channels_last, training):
        """Test behavior in different training modes."""
        layer = GlobalSumPooling2D(**layer_config_channels_last)

        output = layer(sample_input_channels_last, training=training)

        assert output.shape[0] == sample_input_channels_last.shape[0]
        assert output.shape[1] == sample_input_channels_last.shape[3]
        assert len(output.shape) == 2

    def test_compute_output_shape_channels_last(self):
        """Test compute_output_shape method for channels_last."""
        layer = GlobalSumPooling2D(keepdims=False, data_format='channels_last')

        input_shape = (None, 32, 32, 64)
        output_shape = layer.compute_output_shape(input_shape)

        expected_shape = (None, 64)
        assert output_shape == expected_shape

    def test_compute_output_shape_channels_first(self):
        """Test compute_output_shape method for channels_first."""
        layer = GlobalSumPooling2D(keepdims=False, data_format='channels_first')

        input_shape = (None, 64, 32, 32)
        output_shape = layer.compute_output_shape(input_shape)

        expected_shape = (None, 64)
        assert output_shape == expected_shape

    def test_compute_output_shape_keepdims_channels_last(self):
        """Test compute_output_shape method with keepdims=True for channels_last."""
        layer = GlobalSumPooling2D(keepdims=True, data_format='channels_last')

        input_shape = (None, 32, 32, 64)
        output_shape = layer.compute_output_shape(input_shape)

        expected_shape = (None, 1, 1, 64)
        assert output_shape == expected_shape

    def test_compute_output_shape_keepdims_channels_first(self):
        """Test compute_output_shape method with keepdims=True for channels_first."""
        layer = GlobalSumPooling2D(keepdims=True, data_format='channels_first')

        input_shape = (None, 64, 32, 32)
        output_shape = layer.compute_output_shape(input_shape)

        expected_shape = (None, 64, 1, 1)
        assert output_shape == expected_shape

    def test_mathematical_correctness_channels_last(self):
        """Test that the layer computes correct mathematical results for channels_last."""
        # Create a simple test case where we know the expected result
        test_input = keras.ops.ones(shape=(1, 2, 2, 3))  # All ones
        layer = GlobalSumPooling2D(keepdims=False, data_format='channels_last')

        output = layer(test_input)

        # Expected: sum of 2*2=4 ones for each of 3 channels
        expected_output = keras.ops.ones(shape=(1, 3)) * 4.0

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(output),
            keras.ops.convert_to_numpy(expected_output),
            rtol=1e-6, atol=1e-6,
            err_msg="Mathematical result is incorrect"
        )

    def test_mathematical_correctness_channels_first(self):
        """Test that the layer computes correct mathematical results for channels_first."""
        # Create a simple test case where we know the expected result
        test_input = keras.ops.ones(shape=(1, 3, 2, 2))  # All ones
        layer = GlobalSumPooling2D(keepdims=False, data_format='channels_first')

        output = layer(test_input)

        # Expected: sum of 2*2=4 ones for each of 3 channels
        expected_output = keras.ops.ones(shape=(1, 3)) * 4.0

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(output),
            keras.ops.convert_to_numpy(expected_output),
            rtol=1e-6, atol=1e-6,
            err_msg="Mathematical result is incorrect"
        )

    def test_consistency_between_data_formats(self):
        """Test that results are consistent between data formats."""
        # Create equivalent inputs for both formats
        channels_last_input = keras.random.normal(shape=(2, 4, 4, 16), seed=42)
        channels_first_input = keras.ops.transpose(channels_last_input, [0, 3, 1, 2])

        # Create layers for both formats
        layer_cl = GlobalSumPooling2D(keepdims=False, data_format='channels_last')
        layer_cf = GlobalSumPooling2D(keepdims=False, data_format='channels_first')

        # Get outputs
        output_cl = layer_cl(channels_last_input)
        output_cf = layer_cf(channels_first_input)

        # Results should be identical
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(output_cl),
            keras.ops.convert_to_numpy(output_cf),
            rtol=1e-6, atol=1e-6,
            err_msg="Results differ between data formats"
        )

    def test_batch_independence(self, sample_input_channels_last):
        """Test that processing is independent across batch dimension."""
        layer = GlobalSumPooling2D()

        # Process full batch
        full_output = layer(sample_input_channels_last)

        # Process each sample individually
        individual_outputs = []
        for i in range(sample_input_channels_last.shape[0]):
            single_sample = sample_input_channels_last[i:i + 1]
            individual_output = layer(single_sample)
            individual_outputs.append(individual_output)

        # Concatenate individual results
        concatenated_output = keras.ops.concatenate(individual_outputs, axis=0)

        # Results should be identical
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(full_output),
            keras.ops.convert_to_numpy(concatenated_output),
            rtol=1e-6, atol=1e-6,
            err_msg="Batch processing is not independent"
        )

    def test_layer_in_sequential_model(self, sample_input_channels_last):
        """Test layer integration in Sequential model."""
        model = keras.Sequential([
            keras.layers.Conv2D(16, 3, activation='relu', input_shape=sample_input_channels_last.shape[1:]),
            GlobalSumPooling2D(),
            keras.layers.Dense(8, activation='relu'),
            keras.layers.Dense(1)
        ])

        output = model(sample_input_channels_last)

        assert output.shape[0] == sample_input_channels_last.shape[0]
        assert output.shape[1] == 1

    def test_layer_in_functional_model(self, sample_input_channels_last):
        """Test layer integration in functional API model."""
        inputs = keras.Input(shape=sample_input_channels_last.shape[1:])
        x = keras.layers.Conv2D(16, 3, activation='relu')(inputs)
        x = GlobalSumPooling2D()(x)
        x = keras.layers.Dense(8, activation='relu')(x)
        outputs = keras.layers.Dense(1)(x)

        model = keras.Model(inputs, outputs)
        output = model(sample_input_channels_last)

        assert output.shape[0] == sample_input_channels_last.shape[0]
        assert output.shape[1] == 1
