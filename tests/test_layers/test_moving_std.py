import pytest
import numpy as np
import tensorflow as tf
import keras
import tempfile
import os

# Import the layer - adjust the import path as needed
from dl_techniques.layers.statistics.moving_std import MovingStd


class TestMovingStd:
    """Test suite for MovingStd layer implementation."""

    @pytest.fixture
    def input_tensor_channels_last(self):
        """Create a test input tensor with channels_last format."""
        return tf.random.normal([4, 32, 32, 3])

    @pytest.fixture
    def input_tensor_channels_first(self):
        """Create a test input tensor with channels_first format."""
        return tf.random.normal([4, 3, 32, 32])

    @pytest.fixture
    def layer_instance(self):
        """Create a default layer instance for testing."""
        return MovingStd(pool_size=(3, 3))

    def test_initialization_defaults(self):
        """Test initialization with default parameters."""
        layer = MovingStd()

        # Check default values
        assert layer.pool_size == (3, 3)
        assert layer.strides == (1, 1)
        assert layer.padding == "same"
        assert layer.data_format in ["channels_last", "channels_first"]
        assert layer.epsilon == 1e-7
        assert layer.pooler is None

    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        layer = MovingStd(
            pool_size=(5, 5),
            strides=(2, 2),
            padding="valid",
            data_format="channels_first",
            epsilon=1e-5
        )

        # Check custom values
        assert layer.pool_size == (5, 5)
        assert layer.strides == (2, 2)
        assert layer.padding == "valid"
        assert layer.data_format == "channels_first"
        assert layer.epsilon == 1e-5

    def test_invalid_parameters(self):
        """Test that invalid parameters raise appropriate errors."""
        # Invalid pool_size
        with pytest.raises(ValueError, match="pool_size must be a tuple or list of length 2"):
            MovingStd(pool_size=(3,))

        with pytest.raises(ValueError, match="pool_size values must be positive integers"):
            MovingStd(pool_size=(0, 3))

        with pytest.raises(ValueError, match="pool_size values must be positive integers"):
            MovingStd(pool_size=(-1, 3))

        # Invalid strides
        with pytest.raises(ValueError, match="strides must be a tuple or list of length 2"):
            MovingStd(strides=(1,))

        with pytest.raises(ValueError, match="strides values must be positive integers"):
            MovingStd(strides=(0, 1))

        # Invalid padding
        with pytest.raises(ValueError, match="padding must be 'valid' or 'same'"):
            MovingStd(padding="invalid")

        # Invalid data_format
        with pytest.raises(ValueError, match="data_format must be 'channels_first' or 'channels_last'"):
            MovingStd(data_format="invalid")

        # Invalid epsilon
        with pytest.raises(ValueError, match="epsilon must be a non-negative number"):
            MovingStd(epsilon=-1.0)

    def test_build_process(self, input_tensor_channels_last):
        """Test that the layer builds properly."""
        layer = MovingStd(pool_size=(3, 3))

        # Forward pass triggers build
        _ = layer(input_tensor_channels_last)

        # Check that layer was built
        assert layer.built is True
        assert layer.pooler is not None
        assert layer.pooler.built is True
        assert layer._build_input_shape is not None

    def test_build_invalid_input_shape(self):
        """Test that build raises error for invalid input shapes."""
        layer = MovingStd()

        # Test with 3D input (should fail)
        with pytest.raises(ValueError, match="Input must be a 4D tensor"):
            layer.build((None, 32, 32))

    def test_output_shapes_channels_last(self, input_tensor_channels_last):
        """Test output shapes with channels_last format."""
        test_cases = [
            ((3, 3), (1, 1), "same", (4, 32, 32, 3)),
            ((3, 3), (1, 1), "valid", (4, 30, 30, 3)),
            ((5, 5), (2, 2), "same", (4, 16, 16, 3)),
            ((5, 5), (2, 2), "valid", (4, 14, 14, 3)),
        ]

        for pool_size, strides, padding, expected_shape in test_cases:
            layer = MovingStd(
                pool_size=pool_size,
                strides=strides,
                padding=padding,
                data_format="channels_last"
            )
            output = layer(input_tensor_channels_last)

            # Check output shape
            assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"

            # Test compute_output_shape separately
            computed_shape = layer.compute_output_shape(input_tensor_channels_last.shape)
            assert computed_shape == expected_shape

    def test_output_shapes_channels_first(self, input_tensor_channels_first):
        """Test output shapes with channels_first format."""
        test_cases = [
            ((3, 3), (1, 1), "same", (4, 3, 32, 32)),
            ((3, 3), (1, 1), "valid", (4, 3, 30, 30)),
            ((5, 5), (2, 2), "same", (4, 3, 16, 16)),
            ((5, 5), (2, 2), "valid", (4, 3, 14, 14)),
        ]

        for pool_size, strides, padding, expected_shape in test_cases:
            layer = MovingStd(
                pool_size=pool_size,
                strides=strides,
                padding=padding,
                data_format="channels_first"
            )
            output = layer(input_tensor_channels_first)

            # Check output shape
            assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"

            # Test compute_output_shape separately
            computed_shape = layer.compute_output_shape(input_tensor_channels_first.shape)
            assert computed_shape == expected_shape

    def test_forward_pass_correctness(self):
        """Test that forward pass produces mathematically correct results."""
        # Create a simple test case where we can verify the computation
        # Use a small 3x3 input with known values
        input_data = np.array([[[[1.0, 2.0, 3.0],
                                 [4.0, 5.0, 6.0],
                                 [7.0, 8.0, 9.0]]]], dtype=np.float32)

        layer = MovingStd(pool_size=(3, 3), strides=(1, 1), padding="valid")
        output = layer(input_data)

        # For a 3x3 window covering all values, the standard deviation should be:
        # mean = 5.0, variance = ((1-5)^2 + (2-5)^2 + ... + (9-5)^2) / 9
        # variance = (16 + 9 + 4 + 1 + 0 + 1 + 4 + 9 + 16) / 9 = 60/9 = 6.67
        # std = sqrt(6.67) ≈ 2.58

        expected_std = np.sqrt(60.0 / 9.0)

        # Check that output is close to expected (within tolerance)
        assert np.allclose(output.numpy(), expected_std, rtol=1e-5)

    def test_numerical_stability(self):
        """Test layer stability with extreme input values."""
        layer = MovingStd(pool_size=(3, 3), epsilon=1e-7)

        # Test cases with different magnitudes
        test_cases = [
            np.zeros((2, 8, 8, 4), dtype=np.float32),  # All zeros
            np.ones((2, 8, 8, 4), dtype=np.float32) * 1e-10,  # Very small values
            np.ones((2, 8, 8, 4), dtype=np.float32) * 1e10,  # Very large values
            np.random.normal(0, 1e5, (2, 8, 8, 4)).astype(np.float32)  # Large random values
        ]

        for test_input in test_cases:
            output = layer(test_input)

            # Check for NaN/Inf values
            assert not np.any(np.isnan(output.numpy())), "NaN values detected in output"
            assert not np.any(np.isinf(output.numpy())), "Inf values detected in output"

            # Check that output is non-negative (standard deviation is always >= 0)
            assert np.all(output.numpy() >= 0), "Negative values detected in standard deviation"

    def test_different_data_formats(self):
        """Test layer with different data formats."""
        # Test with channels_last
        input_channels_last = tf.random.normal([2, 16, 16, 3])
        layer_channels_last = MovingStd(pool_size=(3, 3), data_format="channels_last")
        output_channels_last = layer_channels_last(input_channels_last)

        # Test with channels_first
        input_channels_first = tf.random.normal([2, 3, 16, 16])
        layer_channels_first = MovingStd(pool_size=(3, 3), data_format="channels_first")
        output_channels_first = layer_channels_first(input_channels_first)

        # Both should produce valid outputs
        assert not np.any(np.isnan(output_channels_last.numpy()))
        assert not np.any(np.isnan(output_channels_first.numpy()))

        # Output shapes should match input format
        assert output_channels_last.shape == (2, 16, 16, 3)
        assert output_channels_first.shape == (2, 3, 16, 16)

    def test_serialization(self):
        """Test serialization and deserialization of the layer."""
        original_layer = MovingStd(
            pool_size=(5, 5),
            strides=(2, 2),
            padding="valid",
            data_format="channels_last",
            epsilon=1e-6
        )

        # Build the layer
        input_shape = (None, 32, 32, 3)
        original_layer.build(input_shape)

        # Get configs
        config = original_layer.get_config()
        build_config = original_layer.get_build_config()

        # Recreate the layer
        recreated_layer = MovingStd.from_config(config)
        recreated_layer.build_from_config(build_config)

        # Check configuration matches
        assert recreated_layer.pool_size == original_layer.pool_size
        assert recreated_layer.strides == original_layer.strides
        assert recreated_layer.padding == original_layer.padding
        assert recreated_layer.data_format == original_layer.data_format
        assert recreated_layer.epsilon == original_layer.epsilon

    def test_model_integration(self, input_tensor_channels_last):
        """Test the layer in a model context."""
        # Create a simple model with the custom layer
        inputs = keras.Input(shape=(32, 32, 3))
        x = MovingStd(pool_size=(3, 3), padding="same")(inputs)
        x = keras.layers.Conv2D(16, (3, 3), activation="relu")(x)
        x = keras.layers.GlobalAveragePooling2D()(x)
        outputs = keras.layers.Dense(10, activation="softmax")(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        # Compile the model
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        # Test forward pass
        y_pred = model(input_tensor_channels_last, training=False)
        assert y_pred.shape == (4, 10)

    def test_model_save_load(self, input_tensor_channels_last):
        """Test saving and loading a model with the custom layer."""
        # Create a model with the custom layer
        inputs = keras.Input(shape=(32, 32, 3))
        x = MovingStd(pool_size=(3, 3), name="moving_std")(inputs)
        x = keras.layers.Conv2D(16, (3, 3))(x)
        x = keras.layers.GlobalAveragePooling2D()(x)
        outputs = keras.layers.Dense(10)(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        # Generate a prediction before saving
        original_prediction = model.predict(input_tensor_channels_last)

        # Create temporary directory for model
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, "model.keras")

            # Save the model
            model.save(model_path)

            # Load the model
            loaded_model = keras.models.load_model(
                model_path,
                custom_objects={"MovingStd": MovingStd}
            )

            # Generate prediction with loaded model
            loaded_prediction = loaded_model.predict(input_tensor_channels_last)

            # Check predictions match
            assert np.allclose(original_prediction, loaded_prediction, rtol=1e-5)

            # Check layer type is preserved
            assert isinstance(loaded_model.get_layer("moving_std"), MovingStd)

    def test_gradient_flow(self, input_tensor_channels_last):
        """Test gradient flow through the layer."""
        layer = MovingStd(pool_size=(3, 3))

        # Create a simple loss function
        with tf.GradientTape() as tape:
            inputs = tf.Variable(input_tensor_channels_last)
            outputs = layer(inputs)
            loss = tf.reduce_mean(tf.square(outputs))

        # Get gradients
        grads = tape.gradient(loss, inputs)

        # Check gradients exist and are not None
        assert grads is not None

        # Check gradients have reasonable values (not all zeros)
        assert not np.allclose(grads.numpy(), 0.0)

    def test_training_behavior(self):
        """Test that the layer behaves correctly during training and inference."""
        layer = MovingStd(pool_size=(3, 3))
        input_data = tf.random.normal([2, 16, 16, 3])

        # Test training=True
        output_train = layer(input_data, training=True)

        # Test training=False
        output_inference = layer(input_data, training=False)

        # For this layer, outputs should be identical regardless of training mode
        assert np.allclose(output_train.numpy(), output_inference.numpy())

    def test_compute_output_shape_without_build(self):
        """Test compute_output_shape method before building the layer."""
        layer = MovingStd(pool_size=(5, 5), strides=(2, 2), padding="valid")

        # Test shape computation without building
        input_shape = (None, 32, 32, 3)
        output_shape = layer.compute_output_shape(input_shape)

        # Expected output shape for these parameters
        expected_shape = (None, 14, 14, 3)
        assert output_shape == expected_shape

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test with minimum pool size
        layer = MovingStd(pool_size=(1, 1))
        input_data = tf.random.normal([1, 4, 4, 1])
        output = layer(input_data)

        # With pool_size=(1,1), standard deviation should be 0 everywhere
        assert np.allclose(output.numpy(), 0.0, atol=1e-6)

        # Test with very small epsilon
        layer_small_eps = MovingStd(pool_size=(3, 3), epsilon=1e-10)
        output_small_eps = layer_small_eps(input_data)
        assert not np.any(np.isnan(output_small_eps.numpy()))

    def test_consistency_across_runs(self):
        """Test that the layer produces consistent outputs across multiple runs."""
        layer = MovingStd(pool_size=(3, 3))

        # Use fixed seed for reproducibility
        tf.random.set_seed(42)
        input_data = tf.random.normal([2, 8, 8, 3])

        # Run multiple times
        outputs = []
        for _ in range(3):
            output = layer(input_data)
            outputs.append(output.numpy())

        # All outputs should be identical
        for i in range(1, len(outputs)):
            assert np.allclose(outputs[0], outputs[i])