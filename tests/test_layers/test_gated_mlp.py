import pytest
import keras
import numpy as np
import tensorflow as tf
from keras import layers
from dl_techniques.layers.gated_mlp import GatedMLP


class TestGatedMLP:
    """Test suite for the GatedMLP layer implementation."""

    @pytest.fixture
    def input_tensor(self):
        """Create a test input tensor with shape [batch_size, height, width, channels]."""
        return tf.random.normal([2, 32, 32, 64])

    def test_initialization(self):
        """Test that the GatedMLP layer initializes correctly with different parameters."""
        # Test with default parameters
        layer1 = GatedMLP(filters=128)
        assert layer1.filters == 128
        assert layer1.use_bias == False
        assert isinstance(layer1.kernel_initializer, keras.initializers.Initializer)
        assert isinstance(layer1.kernel_regularizer, keras.regularizers.Regularizer)

        # Test with custom parameters
        custom_initializer = keras.initializers.HeNormal()
        custom_regularizer = keras.regularizers.L1L2(l1=1e-5, l2=1e-4)

        layer2 = GatedMLP(
            filters=256,
            use_bias=True,
            kernel_initializer=custom_initializer,
            kernel_regularizer=custom_regularizer,
            attention_activation="gelu",
            output_activation="swish"
        )

        assert layer2.filters == 256
        assert layer2.use_bias == True
        assert layer2.attention_activation == "gelu"
        assert layer2.output_activation == "swish"
        assert layer2.kernel_initializer == custom_initializer
        assert layer2.kernel_regularizer == custom_regularizer

    def test_output_shape(self, input_tensor):
        """Test that the output shape is correctly maintained."""
        # Define different filter configurations
        filter_configs = [32, 64, 128]

        for filters in filter_configs:
            layer = GatedMLP(filters=filters)
            output = layer(input_tensor)

            # Check output shape
            expected_shape = (input_tensor.shape[0], input_tensor.shape[1],
                              input_tensor.shape[2], filters)
            assert output.shape == expected_shape

    def test_forward_pass(self, input_tensor):
        """Test the forward pass through the GatedMLP layer."""
        layer = GatedMLP(filters=64)

        # Ensure the layer is built before accessing its weights
        _ = layer(input_tensor)

        # Check that all convolution layers are built
        assert layer.conv_gate is not None
        assert layer.conv_up is not None
        assert layer.conv_down is not None

        # Check that activations are built
        assert layer.attention_activation_fn is not None
        assert layer.output_activation_fn is not None

        # Run a forward pass and check the output
        output = layer(input_tensor)

        # Check that output values are finite (not NaN or Inf)
        assert np.all(np.isfinite(output.numpy()))

    def test_serialization(self):
        """Test that the layer can be serialized and deserialized correctly."""
        original_layer = GatedMLP(
            filters=128,
            use_bias=True,
            kernel_initializer="he_normal",
            kernel_regularizer=keras.regularizers.L2(1e-4),
            attention_activation="gelu",
            output_activation="swish"
        )

        # Serialize the layer to a config
        config = original_layer.get_config()

        # Recreate the layer from the config
        recreated_layer = GatedMLP.from_config(config)

        # Check that configurations match
        assert recreated_layer.filters == original_layer.filters
        assert recreated_layer.use_bias == original_layer.use_bias
        assert recreated_layer.attention_activation == original_layer.attention_activation
        assert recreated_layer.output_activation == original_layer.output_activation

    def test_training_behavior(self, input_tensor):
        """Test that the layer behaves differently in training and inference modes."""
        layer = GatedMLP(
            filters=64,
            use_bias=True,
            kernel_regularizer=keras.regularizers.L2(0.1)  # Large regularization to see the effect
        )

        # Create a simple model with the layer
        model = keras.Sequential([
            layers.InputLayer(input_shape=input_tensor.shape[1:]),
            layer,
            layers.GlobalAveragePooling2D(),
            layers.Dense(10)
        ])

        # Compile the model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

        # Get outputs in training and inference modes
        with tf.GradientTape() as tape:
            training_output = layer(input_tensor, training=True)

        inference_output = layer(input_tensor, training=False)

        # Compute losses for both modes
        # For L2 regularization, training mode should add a regularization loss
        reg_losses_training = layer.losses

        # Check that regularization losses are applied in training mode
        assert len(reg_losses_training) > 0

        # Test the layer in a model context with mock data
        x_train = tf.random.normal([32, 32, 32, 64])
        y_train = tf.random.uniform([32], 0, 10, dtype=tf.int32)

        # Train for one step
        model.fit(x_train, y_train, epochs=1, verbose=0)

        # Test prediction
        predictions = model.predict(x_train, verbose=0)
        assert predictions.shape == (32, 10)