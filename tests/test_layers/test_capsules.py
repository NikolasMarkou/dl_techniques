import pytest
import numpy as np
import keras
import os
import tempfile

from dl_techniques.layers.capsules import (
    SquashLayer,
    PrimaryCapsule,
    RoutingCapsule,
    CapsuleBlock
)

from dl_techniques.utils.tensors import length

class TestSquashLayer:
    """Test suite for SquashLayer implementation."""

    @pytest.fixture
    def input_tensor(self):
        """Create a test input tensor."""
        return keras.random.normal([4, 32, 16])  # batch_size, num_capsules, dim_capsules

    @pytest.fixture
    def layer_instance(self):
        """Create a default layer instance for testing."""
        return SquashLayer()

    def test_initialization_defaults(self):
        """Test initialization with default parameters."""
        layer = SquashLayer()

        # Check default values
        assert layer.axis == -1
        assert layer.epsilon == keras.backend.epsilon()

    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        layer = SquashLayer(axis=1, epsilon=1e-8)

        # Check custom values
        assert layer.axis == 1
        assert layer.epsilon == 1e-8

    def test_build_process(self, input_tensor, layer_instance):
        """Test that the layer builds properly."""
        # Trigger build through forward pass
        output = layer_instance(input_tensor)

        # Check that layer was built
        assert layer_instance.built is True

    def test_output_shapes(self, input_tensor):
        """Test that output shapes are computed correctly."""
        shapes_to_test = [
            (4, 32, 16),
            (2, 10, 8),
            (1, 5, 32),
        ]

        for shape in shapes_to_test:
            layer = SquashLayer()
            test_input = keras.random.normal(shape)
            output = layer(test_input)

            # Check output shape matches input shape
            assert output.shape == test_input.shape

            # Test compute_output_shape separately
            computed_shape = layer.compute_output_shape(test_input.shape)
            assert computed_shape == test_input.shape

    def test_forward_pass(self, input_tensor, layer_instance):
        """Test that forward pass produces expected values."""
        output = layer_instance(input_tensor)

        # Basic sanity checks
        assert not np.any(np.isnan(output.numpy()))
        assert not np.any(np.isinf(output.numpy()))

        # Check output shape
        assert output.shape == input_tensor.shape

        # Check that vector norms are between 0 and 1
        norms = length(output)
        assert np.all(norms.numpy() >= 0.0)
        assert np.all(norms.numpy() <= 1.0)

    def test_different_axes(self):
        """Test squashing along different axes."""
        input_tensor = keras.random.normal([4, 32, 16])

        for axis in [-1, -2, 1, 2]:
            layer = SquashLayer(axis=axis)
            output = layer(input_tensor)

            # Check output is valid
            assert not np.any(np.isnan(output.numpy()))
            assert output.shape == input_tensor.shape

    def test_zero_vectors(self):
        """Test handling of zero vectors."""
        layer = SquashLayer()
        zero_input = keras.ops.zeros([2, 10, 8])
        output = layer(zero_input)

        # Zero vectors should remain zero
        assert np.allclose(output.numpy(), 0.0)

    def test_serialization(self):
        """Test serialization and deserialization of the layer."""
        original_layer = SquashLayer(axis=1, epsilon=1e-8)

        # Build the layer
        input_shape = (None, 32, 16)
        original_layer.build(input_shape)

        # Get configs
        config = original_layer.get_config()
        build_config = original_layer.get_build_config()

        # Recreate the layer
        recreated_layer = SquashLayer.from_config(config)
        recreated_layer.build_from_config(build_config)

        # Check configuration matches
        assert recreated_layer.axis == original_layer.axis
        assert recreated_layer.epsilon == original_layer.epsilon

    def test_numerical_stability(self):
        """Test layer stability with extreme input values."""
        layer = SquashLayer()

        test_cases = [
            keras.ops.ones((2, 10, 8)) * 1e-10,  # Very small values
            keras.ops.ones((2, 10, 8)) * 1e10,  # Very large values
            keras.random.normal((2, 10, 8)) * 1e5  # Large random values
        ]

        for test_input in test_cases:
            output = layer(test_input)

            # Check for NaN/Inf values
            assert not np.any(np.isnan(output.numpy())), "NaN values detected in output"
            assert not np.any(np.isinf(output.numpy())), "Inf values detected in output"

            # Check norms are still valid
            norms = length(output)
            assert np.all(norms.numpy() >= 0.0)
            assert np.all(norms.numpy() <= 1.0001)


class TestPrimaryCapsule:
    """Test suite for PrimaryCapsule implementation."""

    @pytest.fixture
    def input_tensor(self):
        """Create a test input tensor."""
        return keras.random.normal([4, 28, 28, 64])  # batch_size, height, width, channels

    @pytest.fixture
    def layer_instance(self):
        """Create a default layer instance for testing."""
        return PrimaryCapsule(num_capsules=8, dim_capsules=8, kernel_size=9, strides=2)

    def test_initialization_defaults(self):
        """Test initialization with default parameters."""
        layer = PrimaryCapsule(num_capsules=8, dim_capsules=8, kernel_size=9)

        # Check default values
        assert layer.num_capsules == 8
        assert layer.dim_capsules == 8
        assert layer.kernel_size == 9
        assert layer.strides == 1
        assert layer.padding == "valid"
        assert layer.use_bias is True
        assert layer.squash_axis == -1
        assert isinstance(layer.kernel_initializer, keras.initializers.HeNormal)

    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        custom_regularizer = keras.regularizers.L2(1e-4)

        layer = PrimaryCapsule(
            num_capsules=16,
            dim_capsules=4,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="same",
            kernel_initializer="glorot_uniform",
            kernel_regularizer=custom_regularizer,
            use_bias=False,
            squash_axis=-2,
            squash_epsilon=1e-8
        )

        # Check custom values
        assert layer.num_capsules == 16
        assert layer.dim_capsules == 4
        assert layer.kernel_size == (3, 3)
        assert layer.strides == (2, 2)
        assert layer.padding == "same"
        assert layer.use_bias is False
        assert layer.squash_axis == -2
        assert layer.squash_epsilon == 1e-8
        assert isinstance(layer.kernel_initializer, keras.initializers.GlorotUniform)
        assert layer.kernel_regularizer == custom_regularizer

    def test_invalid_parameters(self):
        """Test that invalid parameters raise appropriate errors."""
        # Test negative or zero num_capsules
        with pytest.raises(ValueError, match="num_capsules must be positive"):
            PrimaryCapsule(num_capsules=-8, dim_capsules=8, kernel_size=9)

        with pytest.raises(ValueError, match="num_capsules must be positive"):
            PrimaryCapsule(num_capsules=0, dim_capsules=8, kernel_size=9)

        # Test negative or zero dim_capsules
        with pytest.raises(ValueError, match="dim_capsules must be positive"):
            PrimaryCapsule(num_capsules=8, dim_capsules=-8, kernel_size=9)

        with pytest.raises(ValueError, match="dim_capsules must be positive"):
            PrimaryCapsule(num_capsules=8, dim_capsules=0, kernel_size=9)

        # Test invalid padding
        with pytest.raises(ValueError, match="padding must be one of"):
            PrimaryCapsule(num_capsules=8, dim_capsules=8, kernel_size=9, padding="invalid")

    def test_build_process(self, input_tensor, layer_instance):
        """Test that the layer builds properly."""
        # Trigger build through forward pass
        output = layer_instance(input_tensor)

        # Check that layer was built
        assert layer_instance.built is True
        assert hasattr(layer_instance, "conv")
        assert hasattr(layer_instance, "squash_layer")
        assert layer_instance.conv is not None
        assert layer_instance.squash_layer is not None

    def test_output_shapes(self, input_tensor):
        """Test that output shapes are computed correctly."""
        configs_to_test = [
            {"num_capsules": 8, "dim_capsules": 8, "kernel_size": 9, "strides": 2},
            {"num_capsules": 16, "dim_capsules": 4, "kernel_size": 3, "strides": 1},
            {"num_capsules": 4, "dim_capsules": 16, "kernel_size": 5, "strides": 2},
        ]

        for config in configs_to_test:
            layer = PrimaryCapsule(**config)
            output = layer(input_tensor)

            # Check output has correct number of dimensions
            assert len(output.shape) == 3  # [batch_size, num_total_capsules, dim_capsules]
            assert output.shape[0] == input_tensor.shape[0]  # batch size preserved
            assert output.shape[2] == config["dim_capsules"]  # capsule dimension correct

            # Test compute_output_shape separately
            computed_shape = layer.compute_output_shape(input_tensor.shape)
            assert computed_shape == output.shape

    def test_forward_pass(self, input_tensor, layer_instance):
        """Test that forward pass produces expected values."""
        output = layer_instance(input_tensor)

        # Basic sanity checks
        assert not np.any(np.isnan(output.numpy()))
        assert not np.any(np.isinf(output.numpy()))

        # Check that outputs are properly squashed (norms between 0 and 1)
        norms = length(output)
        assert np.all(norms.numpy() >= 0.0)
        assert np.all(norms.numpy() <= 1.0)

    def test_different_configurations(self):
        """Test layer with different configurations."""
        configurations = [
            {"num_capsules": 8, "dim_capsules": 8, "kernel_size": 9, "strides": 2, "padding": "valid"},
            {"num_capsules": 16, "dim_capsules": 4, "kernel_size": 3, "strides": 1, "padding": "same"},
            {"num_capsules": 32, "dim_capsules": 2, "kernel_size": 5, "strides": 2, "padding": "valid"},
        ]

        for config in configurations:
            layer = PrimaryCapsule(**config)

            # Create test input
            test_input = keras.random.normal([2, 32, 32, 64])

            # Test forward pass
            output = layer(test_input)

            # Check output is valid
            assert not np.any(np.isnan(output.numpy()))
            assert len(output.shape) == 3

    def test_serialization(self):
        """Test serialization and deserialization of the layer."""
        original_layer = PrimaryCapsule(
            num_capsules=16,
            dim_capsules=4,
            kernel_size=3,
            strides=2,
            padding="same",
            kernel_initializer="he_normal",
            use_bias=False,
            squash_epsilon=1e-8
        )

        # Build the layer
        input_shape = (None, 32, 32, 64)
        original_layer.build(input_shape)

        # Get configs
        config = original_layer.get_config()
        build_config = original_layer.get_build_config()

        # Recreate the layer
        recreated_layer = PrimaryCapsule.from_config(config)
        recreated_layer.build_from_config(build_config)

        # Check configuration matches
        assert recreated_layer.num_capsules == original_layer.num_capsules
        assert recreated_layer.dim_capsules == original_layer.dim_capsules
        assert recreated_layer.kernel_size == original_layer.kernel_size
        assert recreated_layer.strides == original_layer.strides
        assert recreated_layer.padding == original_layer.padding
        assert recreated_layer.use_bias == original_layer.use_bias


class TestRoutingCapsule:
    """Test suite for RoutingCapsule implementation."""

    @pytest.fixture
    def input_tensor(self):
        """Create a test input tensor."""
        return keras.random.normal([4, 1152, 8])  # batch_size, num_input_capsules, input_dim_capsules

    @pytest.fixture
    def layer_instance(self):
        """Create a default layer instance for testing."""
        return RoutingCapsule(num_capsules=10, dim_capsules=16, routing_iterations=3)

    def test_initialization_defaults(self):
        """Test initialization with default parameters."""
        layer = RoutingCapsule(num_capsules=10, dim_capsules=16)

        # Check default values
        assert layer.num_capsules == 10
        assert layer.dim_capsules == 16
        assert layer.routing_iterations == 3
        assert layer.use_bias is True
        assert layer.squash_axis == -2
        assert isinstance(layer.kernel_initializer, keras.initializers.GlorotUniform)

    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        custom_regularizer = keras.regularizers.L2(1e-4)

        layer = RoutingCapsule(
            num_capsules=20,
            dim_capsules=32,
            routing_iterations=5,
            kernel_initializer="he_normal",
            kernel_regularizer=custom_regularizer,
            use_bias=False,
            squash_axis=-1,
            squash_epsilon=1e-8
        )

        # Check custom values
        assert layer.num_capsules == 20
        assert layer.dim_capsules == 32
        assert layer.routing_iterations == 5
        assert layer.use_bias is False
        assert layer.squash_axis == -1
        assert layer.squash_epsilon == 1e-8
        assert isinstance(layer.kernel_initializer, keras.initializers.HeNormal)
        assert layer.kernel_regularizer == custom_regularizer

    def test_invalid_parameters(self):
        """Test that invalid parameters raise appropriate errors."""
        # Test negative or zero num_capsules
        with pytest.raises(ValueError, match="num_capsules must be positive"):
            RoutingCapsule(num_capsules=-10, dim_capsules=16)

        with pytest.raises(ValueError, match="num_capsules must be positive"):
            RoutingCapsule(num_capsules=0, dim_capsules=16)

        # Test negative or zero dim_capsules
        with pytest.raises(ValueError, match="dim_capsules must be positive"):
            RoutingCapsule(num_capsules=10, dim_capsules=-16)

        with pytest.raises(ValueError, match="dim_capsules must be positive"):
            RoutingCapsule(num_capsules=10, dim_capsules=0)

        # Test negative or zero routing_iterations
        with pytest.raises(ValueError, match="routing_iterations must be positive"):
            RoutingCapsule(num_capsules=10, dim_capsules=16, routing_iterations=-3)

        with pytest.raises(ValueError, match="routing_iterations must be positive"):
            RoutingCapsule(num_capsules=10, dim_capsules=16, routing_iterations=0)

    def test_build_process(self, input_tensor, layer_instance):
        """Test that the layer builds properly."""
        # Trigger build through forward pass
        output = layer_instance(input_tensor)

        # Check that layer was built
        assert layer_instance.built is True
        assert hasattr(layer_instance, "W")
        assert hasattr(layer_instance, "squash_layer")
        assert layer_instance.W is not None
        assert layer_instance.squash_layer is not None

        # Check weight shapes
        expected_W_shape = (1, input_tensor.shape[1], layer_instance.num_capsules,
                            layer_instance.dim_capsules, input_tensor.shape[2])
        assert layer_instance.W.shape == expected_W_shape

    def test_output_shapes(self, input_tensor):
        """Test that output shapes are computed correctly."""
        configs_to_test = [
            {"num_capsules": 10, "dim_capsules": 16, "routing_iterations": 3},
            {"num_capsules": 5, "dim_capsules": 32, "routing_iterations": 2},
            {"num_capsules": 20, "dim_capsules": 8, "routing_iterations": 4},
        ]

        for config in configs_to_test:
            layer = RoutingCapsule(**config)
            output = layer(input_tensor)

            # Check output shape
            expected_shape = (input_tensor.shape[0], config["num_capsules"], config["dim_capsules"])
            assert output.shape == expected_shape

            # Test compute_output_shape separately
            computed_shape = layer.compute_output_shape(input_tensor.shape)
            assert computed_shape == expected_shape

    def test_forward_pass(self, input_tensor, layer_instance):
        """Test that forward pass produces expected values."""
        output = layer_instance(input_tensor)

        # Basic sanity checks
        assert not np.any(np.isnan(output.numpy()))
        assert not np.any(np.isinf(output.numpy()))

        # Check output shape
        expected_shape = (input_tensor.shape[0], layer_instance.num_capsules, layer_instance.dim_capsules)
        assert output.shape == expected_shape

    def test_different_routing_iterations(self, input_tensor):
        """Test layer with different routing iterations."""
        routing_iterations = [1, 2, 3, 5]

        for iterations in routing_iterations:
            layer = RoutingCapsule(num_capsules=10, dim_capsules=16, routing_iterations=iterations)
            output = layer(input_tensor)

            # Check output is valid
            assert not np.any(np.isnan(output.numpy()))
            assert output.shape == (input_tensor.shape[0], 10, 16)

    def test_serialization(self):
        """Test serialization and deserialization of the layer."""
        original_layer = RoutingCapsule(
            num_capsules=15,
            dim_capsules=24,
            routing_iterations=4,
            kernel_initializer="he_normal",
            use_bias=False,
            squash_epsilon=1e-8
        )

        # Build the layer
        input_shape = (None, 100, 8)
        original_layer.build(input_shape)

        # Get configs
        config = original_layer.get_config()
        build_config = original_layer.get_build_config()

        # Recreate the layer
        recreated_layer = RoutingCapsule.from_config(config)
        recreated_layer.build_from_config(build_config)

        # Check configuration matches
        assert recreated_layer.num_capsules == original_layer.num_capsules
        assert recreated_layer.dim_capsules == original_layer.dim_capsules
        assert recreated_layer.routing_iterations == original_layer.routing_iterations
        assert recreated_layer.use_bias == original_layer.use_bias


class TestCapsuleBlock:
    """Test suite for CapsuleBlock implementation."""

    @pytest.fixture
    def input_tensor(self):
        """Create a test input tensor."""
        return keras.random.normal([4, 1152, 8])  # batch_size, num_input_capsules, input_dim_capsules

    @pytest.fixture
    def layer_instance(self):
        """Create a default layer instance for testing."""
        return CapsuleBlock(num_capsules=10, dim_capsules=16)

    def test_initialization_defaults(self):
        """Test initialization with default parameters."""
        layer = CapsuleBlock(num_capsules=10, dim_capsules=16)

        # Check default values
        assert layer.num_capsules == 10
        assert layer.dim_capsules == 16
        assert layer.routing_iterations == 3
        assert layer.dropout_rate == 0.0
        assert layer.use_layer_norm is False
        assert layer.use_bias is True
        assert isinstance(layer.kernel_initializer, keras.initializers.GlorotUniform)

    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        custom_regularizer = keras.regularizers.L2(1e-4)

        layer = CapsuleBlock(
            num_capsules=20,
            dim_capsules=32,
            routing_iterations=5,
            dropout_rate=0.2,
            use_layer_norm=True,
            kernel_initializer="he_normal",
            kernel_regularizer=custom_regularizer,
            use_bias=False
        )

        # Check custom values
        assert layer.num_capsules == 20
        assert layer.dim_capsules == 32
        assert layer.routing_iterations == 5
        assert layer.dropout_rate == 0.2
        assert layer.use_layer_norm is True
        assert layer.use_bias is False
        assert isinstance(layer.kernel_initializer, keras.initializers.HeNormal)
        assert layer.kernel_regularizer == custom_regularizer

    def test_invalid_parameters(self):
        """Test that invalid parameters raise appropriate errors."""
        # Test invalid dropout rate
        with pytest.raises(ValueError, match="dropout_rate must be in"):
            CapsuleBlock(num_capsules=10, dim_capsules=16, dropout_rate=-0.1)

        with pytest.raises(ValueError, match="dropout_rate must be in"):
            CapsuleBlock(num_capsules=10, dim_capsules=16, dropout_rate=1.1)

        # Test invalid use_layer_norm type
        with pytest.raises(TypeError, match="use_layer_norm must be boolean"):
            CapsuleBlock(num_capsules=10, dim_capsules=16, use_layer_norm="true")

    def test_build_process(self, input_tensor, layer_instance):
        """Test that the layer builds properly."""
        # Trigger build through forward pass
        output = layer_instance(input_tensor)

        # Check that layer was built
        assert layer_instance.built is True
        assert hasattr(layer_instance, "capsule_layer")
        assert layer_instance.capsule_layer is not None

    def test_output_shapes(self, input_tensor):
        """Test that output shapes are computed correctly."""
        configs_to_test = [
            {"num_capsules": 10, "dim_capsules": 16, "dropout_rate": 0.0, "use_layer_norm": False},
            {"num_capsules": 5, "dim_capsules": 32, "dropout_rate": 0.1, "use_layer_norm": True},
            {"num_capsules": 20, "dim_capsules": 8, "dropout_rate": 0.2, "use_layer_norm": False},
        ]

        for config in configs_to_test:
            layer = CapsuleBlock(**config)
            output = layer(input_tensor)

            # Check output shape
            expected_shape = (input_tensor.shape[0], config["num_capsules"], config["dim_capsules"])
            assert output.shape == expected_shape

            # Test compute_output_shape separately
            computed_shape = layer.compute_output_shape(input_tensor.shape)
            assert computed_shape == expected_shape

    def test_forward_pass(self, input_tensor, layer_instance):
        """Test that forward pass produces expected values."""
        output = layer_instance(input_tensor)

        # Basic sanity checks
        assert not np.any(np.isnan(output.numpy()))
        assert not np.any(np.isinf(output.numpy()))

        # Check output shape
        expected_shape = (input_tensor.shape[0], layer_instance.num_capsules, layer_instance.dim_capsules)
        assert output.shape == expected_shape

    def test_with_regularization(self, input_tensor):
        """Test layer with different regularization options."""
        # Test with dropout
        layer_dropout = CapsuleBlock(num_capsules=10, dim_capsules=16, dropout_rate=0.2)
        output_dropout = layer_dropout(input_tensor, training=True)
        assert output_dropout.shape == (input_tensor.shape[0], 10, 16)

        # Test with layer norm
        layer_norm = CapsuleBlock(num_capsules=10, dim_capsules=16, use_layer_norm=True)
        output_norm = layer_norm(input_tensor)
        assert output_norm.shape == (input_tensor.shape[0], 10, 16)

        # Test with both
        layer_both = CapsuleBlock(num_capsules=10, dim_capsules=16, dropout_rate=0.1, use_layer_norm=True)
        output_both = layer_both(input_tensor, training=True)
        assert output_both.shape == (input_tensor.shape[0], 10, 16)

    def test_serialization(self):
        """Test serialization and deserialization of the layer."""
        original_layer = CapsuleBlock(
            num_capsules=15,
            dim_capsules=24,
            routing_iterations=4,
            dropout_rate=0.1,
            use_layer_norm=True,
            kernel_initializer="he_normal",
            use_bias=False
        )

        # Build the layer
        input_shape = (None, 100, 8)
        original_layer.build(input_shape)

        # Get configs
        config = original_layer.get_config()
        build_config = original_layer.get_build_config()

        # Recreate the layer
        recreated_layer = CapsuleBlock.from_config(config)
        recreated_layer.build_from_config(build_config)

        # Check configuration matches
        assert recreated_layer.num_capsules == original_layer.num_capsules
        assert recreated_layer.dim_capsules == original_layer.dim_capsules
        assert recreated_layer.routing_iterations == original_layer.routing_iterations
        assert recreated_layer.dropout_rate == original_layer.dropout_rate
        assert recreated_layer.use_layer_norm == original_layer.use_layer_norm
        assert recreated_layer.use_bias == original_layer.use_bias

    def test_model_integration(self, input_tensor):
        """Test the layer in a model context."""
        # Create a simple model with the capsule block
        inputs = keras.Input(shape=input_tensor.shape[1:])
        x = CapsuleBlock(num_capsules=10, dim_capsules=16)(inputs)
        x = keras.layers.Lambda(lambda x: length(x))(x)  # Get capsule lengths
        outputs = keras.layers.Dense(10, activation='softmax')(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        # Compile the model
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
        )

        # Test forward pass
        y_pred = model(input_tensor, training=False)
        assert y_pred.shape == (input_tensor.shape[0], 10)

    def test_model_save_load(self, input_tensor):
        """Test saving and loading a model with the capsule block."""
        # Create a model with the capsule block
        inputs = keras.Input(shape=input_tensor.shape[1:])
        x = CapsuleBlock(num_capsules=10, dim_capsules=16, name="capsule_block")(inputs)
        outputs = keras.layers.Dense(10, activation='softmax')(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        # Generate a prediction before saving
        original_prediction = model.predict(input_tensor, verbose=0)

        # Create temporary directory for model
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, "model.keras")

            # Save the model
            model.save(model_path)

            # Load the model
            loaded_model = keras.models.load_model(
                model_path,
                custom_objects={
                    "CapsuleBlock": CapsuleBlock,
                    "RoutingCapsule": RoutingCapsule,
                    "SquashLayer": SquashLayer,
                }
            )

            # Generate prediction with loaded model
            loaded_prediction = loaded_model.predict(input_tensor, verbose=0)

            # Check predictions match
            assert np.allclose(original_prediction, loaded_prediction, rtol=1e-5)

            # Check layer type is preserved
            assert isinstance(loaded_model.get_layer("capsule_block"), CapsuleBlock)

    def test_numerical_stability(self):
        """Test layer stability with extreme input values."""
        layer = CapsuleBlock(num_capsules=5, dim_capsules=8)

        # Create inputs with different magnitudes
        batch_size = 2
        num_input_capsules = 100
        input_dim_capsules = 4

        test_cases = [
            keras.ops.zeros((batch_size, num_input_capsules, input_dim_capsules)),  # Zeros
            keras.ops.ones((batch_size, num_input_capsules, input_dim_capsules)) * 1e-10,  # Very small values
            keras.ops.ones((batch_size, num_input_capsules, input_dim_capsules)) * 1e5,  # Large values
            keras.random.normal((batch_size, num_input_capsules, input_dim_capsules)) * 100  # Large random values
        ]

        for test_input in test_cases:
            output = layer(test_input)

            # Check for NaN/Inf values
            assert not np.any(np.isnan(output.numpy())), "NaN values detected in output"
            assert not np.any(np.isinf(output.numpy())), "Inf values detected in output"

    def test_regularization(self, input_tensor):
        """Test that regularization losses are properly applied."""
        # Create layer with regularization
        layer = CapsuleBlock(
            num_capsules=10,
            dim_capsules=16,
            kernel_regularizer=keras.regularizers.L2(0.1)
        )

        # Build layer
        layer.build(input_tensor.shape)

        # No regularization losses before calling the layer
        initial_losses = len(layer.losses)

        # Apply the layer
        _ = layer(input_tensor)

        # Should have regularization losses now
        assert len(layer.losses) >= initial_losses


class TestUtilityFunctions:
    """Test suite for utility functions."""

    def test_length_function(self):
        """Test the length utility function."""
        # Test with known vectors
        vectors = keras.ops.array([
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],  # Unit vectors
            [[3.0, 4.0, 0.0], [0.0, 0.0, 5.0]]  # Known lengths: 5, 5
        ])

        lengths = length(vectors)
        expected_lengths = keras.ops.array([
            [1.0, 1.0],
            [5.0, 5.0]
        ])

        assert np.allclose(lengths.numpy(), expected_lengths.numpy(), rtol=1e-6)

    def test_length_numerical_stability(self):
        """Test length function with extreme values."""
        # Very small vectors
        small_vectors = keras.ops.ones((2, 3, 4)) * 1e-10
        small_lengths = length(small_vectors)
        assert not np.any(np.isnan(small_lengths.numpy()))
        assert not np.any(np.isinf(small_lengths.numpy()))

        # Very large vectors
        large_vectors = keras.ops.ones((2, 3, 4)) * 1e10
        large_lengths = length(large_vectors)
        assert not np.any(np.isnan(large_lengths.numpy()))
        assert not np.any(np.isinf(large_lengths.numpy()))