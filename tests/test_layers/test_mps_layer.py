"""Comprehensive test suite for the MPSLayer.

This module contains pytests for the MPSLayer implementation, covering
initialization, build process, shape computation, forward pass correctness,
serialization, and model integration.
"""

import pytest
import numpy as np
import keras
from keras import ops
import tensorflow as tf
import tempfile
import os

# Adjust the import path as necessary for your project structure.
from dl_techniques.layers.mps_layer import MPSLayer


class TestMPSLayer:
    """Test suite for the MPSLayer implementation."""

    @pytest.fixture
    def input_tensor(self):
        """Create a standard 2D input tensor."""
        # Batch=8, input_dim=16
        return keras.random.normal([8, 16])

    @pytest.fixture
    def layer_instance(self):
        """Create a default layer instance for testing."""
        return MPSLayer(output_dim=32, bond_dim=8)

    def test_initialization_defaults(self):
        """Test initialization with default parameters."""
        layer = MPSLayer(output_dim=64)
        assert layer.output_dim == 64
        assert layer.bond_dim == 16
        assert layer.use_bias is True
        assert isinstance(layer.kernel_initializer, keras.initializers.GlorotUniform)
        assert isinstance(layer.bias_initializer, keras.initializers.Zeros)
        assert layer.kernel_regularizer is None
        assert layer.bias_regularizer is None
        assert layer.activity_regularizer is None
        assert layer.built is False

    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        layer = MPSLayer(
            output_dim=32,
            bond_dim=8,
            use_bias=False,
            kernel_initializer="he_normal",
            kernel_regularizer="l1",
            name="custom_mps"
        )
        assert layer.output_dim == 32
        assert layer.bond_dim == 8
        assert layer.use_bias is False
        assert layer.name == "custom_mps"
        assert isinstance(layer.kernel_initializer, keras.initializers.HeNormal)
        assert isinstance(layer.kernel_regularizer, keras.regularizers.L1)
        assert layer.bias_weight is None

    def test_invalid_parameters(self):
        """Test that invalid parameters raise appropriate errors."""
        with pytest.raises(ValueError, match="output_dim must be positive"):
            MPSLayer(output_dim=0)
        with pytest.raises(ValueError, match="output_dim must be positive"):
            MPSLayer(output_dim=-10)
        with pytest.raises(ValueError, match="bond_dim must be positive"):
            MPSLayer(output_dim=32, bond_dim=0)
        with pytest.raises(ValueError, match="bond_dim must be positive"):
            MPSLayer(output_dim=32, bond_dim=-8)

    def test_build_process(self, input_tensor, layer_instance):
        """Test that the layer builds properly and creates correct weights."""
        assert layer_instance.built is False
        layer_instance(input_tensor)  # Trigger build
        assert layer_instance.built is True

        batch_size, input_dim = input_tensor.shape

        # Check that weights are created and have correct shapes
        assert layer_instance.cores is not None
        assert layer_instance.projection is not None
        assert layer_instance.bias_weight is not None

        assert layer_instance.cores.shape == (input_dim, layer_instance.bond_dim, layer_instance.bond_dim)
        assert layer_instance.projection.shape == (layer_instance.bond_dim, layer_instance.output_dim)
        assert layer_instance.bias_weight.shape == (layer_instance.output_dim,)

        # Check number of trainable weights
        assert len(layer_instance.trainable_weights) == 3

    def test_build_with_no_bias(self, input_tensor):
        """Test the build process when use_bias=False."""
        layer = MPSLayer(output_dim=32, bond_dim=8, use_bias=False)
        layer(input_tensor) # Trigger build

        assert layer.built is True
        assert layer.bias_weight is None
        assert len(layer.trainable_weights) == 2

    @pytest.mark.parametrize("output_dim, bond_dim", [(64, 16), (1, 2), (128, 4)])
    def test_output_shapes(self, input_tensor, output_dim, bond_dim):
        """Test that output shapes are computed correctly."""
        layer = MPSLayer(output_dim=output_dim, bond_dim=bond_dim)
        output = layer(input_tensor)

        expected_shape = (input_tensor.shape[0], output_dim)
        assert output.shape == expected_shape

        computed_shape = layer.compute_output_shape(input_tensor.shape)
        assert computed_shape == expected_shape

    def test_output_shapes_with_unknown_batch(self):
        """Test compute_output_shape with an unknown batch size."""
        layer = MPSLayer(output_dim=32, bond_dim=8)
        input_shape = (None, 64)
        output_shape = layer.compute_output_shape(input_shape)
        assert output_shape == (None, 32)

    def test_forward_pass(self, input_tensor, layer_instance):
        """Test the forward pass for correct shape and validity."""
        output = layer_instance(input_tensor)
        assert output.shape == (input_tensor.shape[0], layer_instance.output_dim)
        assert not np.any(np.isnan(output.numpy()))
        assert not np.any(np.isinf(output.numpy()))

    def test_bias_effect(self, input_tensor):
        """Test that the bias is correctly added to the output."""
        # Use a non-zero initializer to see the effect
        layer_with_bias = MPSLayer(output_dim=32, bond_dim=8, use_bias=True, bias_initializer='ones')
        layer_no_bias = MPSLayer(output_dim=32, bond_dim=8, use_bias=False)

        # Ensure kernel weights are the same for a fair comparison
        layer_with_bias.build(input_tensor.shape)
        layer_no_bias.build(input_tensor.shape)
        layer_no_bias.set_weights(layer_with_bias.get_weights()[:-1]) # All but bias

        output_with_bias = layer_with_bias(input_tensor)
        output_no_bias = layer_no_bias(input_tensor)

        bias_vector = layer_with_bias.bias_weight

        # The output should be the no-bias output plus the bias vector
        assert np.allclose(output_with_bias.numpy(), output_no_bias.numpy() + bias_vector.numpy(), atol=1e-6)

    def test_regularizers(self, input_tensor):
        """Test that kernel and bias regularizers add losses."""
        layer = MPSLayer(
            output_dim=32, bond_dim=4,
            kernel_regularizer=keras.regularizers.L2(0.01),
            bias_regularizer=keras.regularizers.L1(0.01),
        )
        layer(input_tensor)  # Build layer
        # **FIXED**: Expect 3 losses: one for `cores`, one for `projection` (both
        # using kernel_regularizer), and one for `bias`.
        assert len(layer.losses) == 3
        # The L1 regularizer on the zero-initialized bias might be zero.
        # We check that the kernel regularizer losses are non-zero.
        assert layer.losses[0] > 0
        assert layer.losses[1] > 0

    def test_serialization(self, input_tensor):
        """Test serialization and deserialization of the layer."""
        original_layer = MPSLayer(
            output_dim=32,
            bond_dim=8,
            use_bias=True,
            kernel_initializer="ones",
            name="serial_mps"
        )
        output_original = original_layer(input_tensor)

        config = original_layer.get_config()
        recreated_layer = MPSLayer.from_config(config)

        # Check configuration
        assert recreated_layer.output_dim == original_layer.output_dim
        assert recreated_layer.bond_dim == original_layer.bond_dim
        assert recreated_layer.use_bias == original_layer.use_bias
        assert isinstance(recreated_layer.kernel_initializer, keras.initializers.Ones)

        # Check output after rebuilding with same weights
        recreated_layer.build(input_tensor.shape)
        recreated_layer.set_weights(original_layer.get_weights())
        output_recreated = recreated_layer(input_tensor)

        assert np.allclose(output_original.numpy(), output_recreated.numpy(), atol=1e-6)

    def test_model_save_load(self, input_tensor):
        """Test saving and loading a model with the MPSLayer."""
        inputs = keras.Input(shape=input_tensor.shape[1:])
        x = MPSLayer(output_dim=32, bond_dim=8, name="mps_in_model")(inputs)
        outputs = keras.layers.Dense(10)(x)
        model = keras.Model(inputs=inputs, outputs=outputs)

        original_prediction = model.predict(input_tensor, verbose=0)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.keras")
            model.save(model_path)

            loaded_model = keras.models.load_model(
                model_path, custom_objects={"MPSLayer": MPSLayer}
            )

            loaded_prediction = loaded_model.predict(input_tensor, verbose=0)
            assert np.allclose(original_prediction, loaded_prediction, rtol=1e-5)

            # Check layer type and config are preserved
            mps_layer = loaded_model.get_layer("mps_in_model")
            assert isinstance(mps_layer, MPSLayer)
            assert mps_layer.output_dim == 32
            assert mps_layer.bond_dim == 8

    def test_gradient_flow(self, input_tensor):
        """Test that gradients flow properly through the layer."""
        layer = MPSLayer(output_dim=32, bond_dim=8)

        input_var = tf.Variable(input_tensor)

        with tf.GradientTape() as tape:
            output = layer(input_var)
            loss = ops.mean(output**2)

        # Gradients w.r.t trainable weights
        trainable_vars = layer.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        assert len(gradients) == len(trainable_vars)
        for grad, var in zip(gradients, trainable_vars):
            assert grad is not None
            assert grad.shape == var.shape
            assert not np.any(np.isnan(grad.numpy()))

    def test_training_compatibility(self):
        """Test that the layer works correctly during training."""
        model = keras.Sequential([
            keras.layers.Input(shape=(32,)),
            MPSLayer(output_dim=16, bond_dim=4),
            keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')

        x_train = keras.random.normal([64, 32])
        y_train = keras.random.normal([64, 1])

        history = model.fit(x_train, y_train, epochs=2, batch_size=16, verbose=0)

        # Training should complete without errors, and loss should change
        assert len(history.history['loss']) == 2
        assert history.history['loss'][0] != history.history['loss'][1]