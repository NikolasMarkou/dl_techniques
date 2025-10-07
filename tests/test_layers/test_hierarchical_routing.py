import pytest
import numpy as np
import tensorflow as tf
import keras
import os
import tempfile

from dl_techniques.layers.hierarchical_routing import HierarchicalRoutingLayer

# ==============================================================================
# Test Suite
# ==============================================================================


class TestHierarchicalRoutingLayer:
    """
    Test suite for HierarchicalRoutingLayer implementation.

    This suite uses pytest.mark.parametrize to test the layer against a diverse
    set of input and output dimensions for enhanced robustness.
    """

    @pytest.mark.parametrize(
        "output_dim, expected_padded, expected_decisions",
        [
            (2, 2, 1),          # Minimum valid dimension (power of 2)
            (3, 4, 2),          # Small non-power-of-2
            (7, 8, 3),          # Non-power-of-2 just under a power of 2
            (8, 8, 3),          # Power of 2
            (17, 32, 5),        # Non-power-of-2 just over a power of 2
            (1000, 1024, 10),   # Larger, more realistic dimension
        ]
    )
    def test_initialization(self, output_dim, expected_padded, expected_decisions):
        """Test initialization of the layer with diverse output dimensions."""
        layer = HierarchicalRoutingLayer(output_dim=output_dim)
        assert isinstance(layer, keras.layers.Layer)
        assert layer.output_dim == output_dim
        assert layer.padded_output_dim == expected_padded
        assert layer.num_decisions == expected_decisions
        assert isinstance(layer.decision_dense, keras.layers.Dense)
        assert layer.decision_dense.units == expected_decisions

    def test_invalid_initialization(self):
        """Test that invalid parameters raise appropriate errors."""
        with pytest.raises(ValueError, match="must be an integer greater than 1"):
            HierarchicalRoutingLayer(output_dim=1)
        with pytest.raises(ValueError, match="must be an integer greater than 1"):
            HierarchicalRoutingLayer(output_dim=0)

    @pytest.mark.parametrize("batch_size, input_features", [(1, 8), (4, 16), (32, 64)])
    def test_build_process(self, batch_size, input_features):
        """Test that the layer and its sub-layers build properly."""
        output_dim = 10
        num_decisions = 4  # log2(16)
        layer = HierarchicalRoutingLayer(output_dim=output_dim)
        inputs = tf.random.normal([batch_size, input_features])

        assert not layer.built
        layer(inputs)  # Forward pass triggers build
        assert layer.built
        assert layer.decision_dense.built
        assert len(layer.trainable_weights) == 1
        assert layer.weights[0].shape == (input_features, num_decisions)

    @pytest.mark.parametrize(
        "batch_size, input_features, output_dim",
        [
            (1, 8, 5),          # Small and simple
            (4, 16, 10),        # Original test case
            (32, 64, 128),      # Larger, all powers of 2
            (7, 21, 33),        # Awkward, non-power-of-2 dimensions
        ]
    )
    def test_output_shapes(self, batch_size, input_features, output_dim):
        """Test that output shapes are computed correctly across diverse dimensions."""
        layer = HierarchicalRoutingLayer(output_dim=output_dim)
        inputs = tf.random.normal([batch_size, input_features])
        output = layer(inputs)

        expected_shape = (batch_size, output_dim)
        assert output.shape == expected_shape
        computed_shape = layer.compute_output_shape(inputs.shape)
        assert computed_shape == expected_shape

    @pytest.mark.parametrize(
        "batch_size, input_features, output_dim",
        [(1, 8, 5), (4, 16, 10), (32, 64, 128), (7, 21, 33)]
    )
    def test_output_is_normalized(self, batch_size, input_features, output_dim):
        """Test that the output is a valid probability distribution (sums to 1)."""
        layer = HierarchicalRoutingLayer(output_dim=output_dim)
        inputs = tf.random.normal([batch_size, input_features])
        output = layer(inputs)

        sums = tf.reduce_sum(output, axis=1).numpy()
        assert np.allclose(sums, 1.0, atol=1e-6)

    def test_probability_calculation_manual_non_power_of_2(self):
        """Test the routing logic for a non-power-of-2 case requiring renormalization."""
        input_dim, output_dim, epsilon = 2, 6, 1e-7
        layer = HierarchicalRoutingLayer(output_dim=output_dim, use_bias=True, epsilon=epsilon)
        inputs = tf.ones((1, input_dim))
        layer(inputs)  # Build layer

        kernel = np.zeros((input_dim, layer.num_decisions), dtype=np.float32)
        bias = np.array([100.0, -100.0, 0.0], dtype=np.float32)
        layer.decision_dense.set_weights([kernel, bias])

        d1, d2, d3 = 1.0 - epsilon, epsilon, 0.5
        expected_padded = np.array([
            (1 - d1) * (1 - d2) * (1 - d3), (1 - d1) * (1 - d2) * d3,
            (1 - d1) * d2 * (1 - d3), (1 - d1) * d2 * d3,
            d1 * (1 - d2) * (1 - d3), d1 * (1 - d2) * d3,
            d1 * d2 * (1 - d3), d1 * d2 * d3,
        ])

        unnormalized = expected_padded[:output_dim]
        expected_final = unnormalized / np.sum(unnormalized)
        output = layer(inputs).numpy().flatten()
        assert np.allclose(output, expected_final, atol=1e-6)

    def test_probability_calculation_manual_power_of_2(self):
        """Test the routing logic for a power-of-2 case without renormalization."""
        input_dim, output_dim, epsilon = 2, 4, 1e-7
        layer = HierarchicalRoutingLayer(output_dim=output_dim, use_bias=True, epsilon=epsilon)
        inputs = tf.ones((1, input_dim))
        layer(inputs)  # Build layer (num_decisions = 2)

        kernel = np.zeros((input_dim, layer.num_decisions), dtype=np.float32)
        bias = np.array([100.0, -100.0], dtype=np.float32)
        layer.decision_dense.set_weights([kernel, bias])

        d1, d2 = 1.0 - epsilon, epsilon
        expected_final = np.array([
            (1 - d1) * (1 - d2), (1 - d1) * d2,
            d1 * (1 - d2), d1 * d2,
        ])

        output = layer(inputs).numpy().flatten()
        assert np.allclose(output, expected_final, atol=1e-6)

    @pytest.mark.parametrize("output_dim", [2, 10, 32])
    def test_stability_with_extreme_inputs(self, output_dim):
        """Test that the layer does not produce exact zeros, preventing NaN loss."""
        layer = HierarchicalRoutingLayer(output_dim=output_dim, use_bias=True)
        inputs = tf.ones((1, 2))
        layer(inputs)

        bias = np.tile([-100.0, 100.0], layer.num_decisions // 2 + 1)[:layer.num_decisions]
        kernel = np.zeros((2, layer.num_decisions), dtype=np.float32)
        layer.decision_dense.set_weights([kernel, bias])

        output_probs = layer(inputs).numpy().flatten()
        assert np.all(output_probs > 0), "Output contains zero probabilities"
        assert not np.any(np.isnan(output_probs)), "Output contains NaN values"

    def test_serialization(self):
        """Test serialization and deserialization of the layer."""
        original_layer = HierarchicalRoutingLayer(output_dim=12, epsilon=1e-6, name="test_routing")
        config = original_layer.get_config()
        recreated_layer = HierarchicalRoutingLayer.from_config(config)

        assert recreated_layer.name == original_layer.name
        assert recreated_layer.output_dim == original_layer.output_dim
        assert recreated_layer.epsilon == original_layer.epsilon
        assert recreated_layer.padded_output_dim == 16

    @pytest.mark.parametrize("input_features, output_dim", [(16, 10), (64, 33)])
    def test_model_save_load(self, input_features, output_dim):
        """Test saving and loading a model with the HierarchicalRoutingLayer."""
        inputs = keras.Input(shape=(input_features,))
        x = keras.layers.Dense(32, activation='relu')(inputs)
        outputs = HierarchicalRoutingLayer(output_dim=output_dim, name="routing_output")(x)
        model = keras.Model(inputs=inputs, outputs=outputs)

        input_data = tf.random.normal([4, input_features])
        original_prediction = model.predict(input_data)

        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, "model.keras")
            model.save(model_path)
            loaded_model = keras.models.load_model(model_path)
            loaded_prediction = loaded_model.predict(input_data)

            assert np.allclose(original_prediction, loaded_prediction, atol=1e-6)
            assert isinstance(loaded_model.get_layer("routing_output"), HierarchicalRoutingLayer)

    def test_gradient_flow(self):
        """Test gradient flow through the layer."""
        inputs = tf.random.normal([4, 16])
        layer = HierarchicalRoutingLayer(output_dim=10)

        with tf.GradientTape() as tape:
            tape.watch(inputs)
            outputs = layer(inputs)
            loss = tf.reduce_mean(tf.square(outputs))

        grads = tape.gradient(loss, layer.trainable_weights)
        assert len(grads) > 0, "No gradients were computed."
        for grad in grads:
            assert grad is not None, "A gradient is None."
            assert not np.all(grad.numpy() == 0), "Gradients are all zero."

    @pytest.mark.parametrize(
        "batch_size, input_dim, output_dim",
        [
            (16, 8, 5),     # Original case
            (32, 32, 17),   # Larger, non-power-of-2 case
            (8, 16, 2),     # Minimum output dimension
        ]
    )
    def test_training_loop(self, batch_size, input_dim, output_dim):
        """Test that the layer can be used in a training loop with diverse shapes."""
        model = keras.Sequential([
            keras.Input(shape=(input_dim,)),
            keras.layers.Dense(16, activation='relu'),
            HierarchicalRoutingLayer(output_dim=output_dim)
        ])
        model.compile(optimizer="adam", loss="categorical_crossentropy")

        x_train = tf.random.normal([batch_size, input_dim])
        y_train_indices = tf.random.uniform([batch_size], 0, output_dim, dtype=tf.int32)
        y_train = tf.one_hot(y_train_indices, depth=output_dim)

        initial_loss = model.evaluate(x_train, y_train, verbose=0)
        model.fit(x_train, y_train, epochs=2, batch_size=4, verbose=0)
        final_loss = model.evaluate(x_train, y_train, verbose=0)

        assert final_loss < initial_loss
        assert not np.isnan(final_loss), "Loss became NaN during training"