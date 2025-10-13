import pytest
import numpy as np
import tensorflow as tf
import keras
import os
import tempfile

from dl_techniques.layers.routing_probabilities import RoutingProbabilitiesLayer


# ==============================================================================
# Test Suite
# ==============================================================================


class TestHierarchicalRoutingBasicLayer:
    """
    Test suite for the non-trainable HierarchicalRoutingBasicLayer implementation.

    This suite tests the deterministic, parameter-free hierarchical routing layer
    that acts as a drop-in replacement for softmax activation.
    """

    @pytest.mark.parametrize(
        "output_dim, expected_padded, expected_decisions",
        [
            (2, 2, 1),  # Minimum valid dimension (power of 2)
            (3, 4, 2),  # Small non-power-of-2
            (7, 8, 3),  # Non-power-of-2 just under a power of 2
            (8, 8, 3),  # Power of 2
            (17, 32, 5),  # Non-power-of-2 just over a power of 2
            (1000, 1024, 10),  # Larger, more realistic dimension
        ]
    )
    def test_initialization_with_explicit_output_dim(
            self, output_dim, expected_padded, expected_decisions
    ):
        """Test initialization with explicit output_dim parameter."""
        layer = RoutingProbabilitiesLayer(output_dim=output_dim)
        assert isinstance(layer, keras.layers.Layer)
        assert layer.output_dim == output_dim
        # Before build, these should be None
        assert layer.padded_output_dim is None
        assert layer.num_decisions is None
        assert layer.decision_weights is None

        # After build
        inputs = tf.random.normal([1, 16])
        layer(inputs)
        assert layer.padded_output_dim == expected_padded
        assert layer.num_decisions == expected_decisions
        assert layer.decision_weights is not None

    def test_initialization_without_output_dim(self):
        """Test initialization without output_dim (to be inferred)."""
        layer = RoutingProbabilitiesLayer()
        assert layer.output_dim is None
        assert layer.padded_output_dim is None
        assert layer.num_decisions is None

        # Build with specific input shape
        inputs = tf.random.normal([1, 10])
        output = layer(inputs)

        # output_dim should be inferred as 10
        assert layer.output_dim == 10
        assert layer.padded_output_dim == 16  # Next power of 2
        assert layer.num_decisions == 4
        assert output.shape == (1, 10)

    def test_invalid_initialization(self):
        """Test that invalid parameters raise appropriate errors."""
        with pytest.raises(ValueError, match="must be an integer greater than 1"):
            RoutingProbabilitiesLayer(output_dim=1)
        with pytest.raises(ValueError, match="must be an integer greater than 1"):
            RoutingProbabilitiesLayer(output_dim=0)
        with pytest.raises(ValueError, match="must be an integer greater than 1"):
            layer = RoutingProbabilitiesLayer(output_dim=-5)

    def test_invalid_inference_none_input_dim(self):
        """Test that build fails when trying to infer from None dimension."""
        layer = RoutingProbabilitiesLayer()  # output_dim not provided

        with pytest.raises(ValueError, match="Cannot infer output_dim"):
            # Try to build with unknown last dimension
            layer.build((None, None))

    @pytest.mark.parametrize("batch_size, input_features", [(1, 8), (4, 16), (32, 64)])
    def test_build_process(self, batch_size, input_features):
        """Test that the layer builds properly without trainable weights."""
        output_dim = 10
        layer = RoutingProbabilitiesLayer(output_dim=output_dim)
        inputs = tf.random.normal([batch_size, input_features])

        assert not layer.built
        output = layer(inputs)
        assert layer.built

        # Critical: Layer should have NO trainable weights
        assert len(layer.trainable_weights) == 0
        # It has no non-trainable weights as they are recomputed on build.
        assert len(layer.non_trainable_weights) == 0

        # Decision weights should be precomputed
        assert layer.decision_weights is not None
        assert layer.decision_weights.shape == (layer.num_decisions, input_features)

    @pytest.mark.parametrize(
        "batch_size, input_features, output_dim",
        [
            (1, 8, 5),  # Small and simple
            (4, 16, 10),  # Original test case
            (32, 64, 128),  # Larger, all powers of 2
            (7, 21, 33),  # Awkward, non-power-of-2 dimensions
        ]
    )
    def test_output_shapes(self, batch_size, input_features, output_dim):
        """Test that output shapes are computed correctly across diverse dimensions."""
        layer = RoutingProbabilitiesLayer(output_dim=output_dim)
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
        layer = RoutingProbabilitiesLayer(output_dim=output_dim)
        inputs = tf.random.normal([batch_size, input_features])
        output = layer(inputs)

        sums = tf.reduce_sum(output, axis=1).numpy()
        assert np.allclose(sums, 1.0, atol=1e-6), f"Probabilities don't sum to 1: {sums}"

    @pytest.mark.parametrize("output_dim", [2, 5, 10, 32, 100])
    def test_output_values_in_valid_range(self, output_dim):
        """Test that all output probabilities are in [0, 1] range."""
        layer = RoutingProbabilitiesLayer(output_dim=output_dim)
        inputs = tf.random.normal([16, 32])
        output = layer(inputs).numpy()

        assert np.all(output >= 0.0), "Found negative probabilities"
        assert np.all(output <= 1.0), "Found probabilities > 1"
        assert np.all(output > 0.0), "Found zero probabilities (should be > epsilon)"

    def test_deterministic_behavior(self):
        """Test that the layer produces deterministic outputs (no randomness)."""
        layer = RoutingProbabilitiesLayer(output_dim=10)
        inputs = tf.constant([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]])

        # Run multiple times with same input
        output_1 = layer(inputs)
        output_2 = layer(inputs)
        output_3 = layer(inputs)

        # All outputs should be identical
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(output_1),
            keras.ops.convert_to_numpy(output_2),
            rtol=1e-6, atol=1e-6,
            err_msg="Outputs should be deterministic"
        )
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(output_2),
            keras.ops.convert_to_numpy(output_3),
            rtol=1e-6, atol=1e-6,
            err_msg="Outputs should be deterministic"
        )

    def test_different_inputs_produce_different_outputs(self):
        """Test that different inputs produce different probability distributions."""
        layer = RoutingProbabilitiesLayer(output_dim=10)

        inputs_1 = tf.constant([[1.0, 2.0, 3.0, 4.0, 5.0]])
        inputs_2 = tf.constant([[5.0, 4.0, 3.0, 2.0, 1.0]])

        output_1 = layer(inputs_1).numpy()
        output_2 = layer(inputs_2).numpy()

        # Outputs should be different
        assert not np.allclose(output_1, output_2, atol=1e-3), \
            "Different inputs should produce different outputs"

    def test_power_of_2_output_dim(self):
        """Test handling of power-of-2 output dimensions (no renormalization)."""
        for output_dim in [2, 4, 8, 16, 32]:
            layer = RoutingProbabilitiesLayer(output_dim=output_dim)
            inputs = tf.random.normal([8, 16])
            output = layer(inputs)

            assert output.shape == (8, output_dim)
            # Should still be normalized
            sums = tf.reduce_sum(output, axis=1).numpy()
            assert np.allclose(sums, 1.0, atol=1e-6)

    def test_non_power_of_2_output_dim(self):
        """Test handling of non-power-of-2 output dimensions (with renormalization)."""
        for output_dim in [3, 5, 7, 13, 17, 100]:
            layer = RoutingProbabilitiesLayer(output_dim=output_dim)
            inputs = tf.random.normal([8, 16])
            output = layer(inputs)

            assert output.shape == (8, output_dim)
            # Should be normalized after slicing and renormalization
            sums = tf.reduce_sum(output, axis=1).numpy()
            assert np.allclose(sums, 1.0, atol=1e-6)

    @pytest.mark.parametrize("output_dim", [2, 10, 32, 100])
    def test_stability_with_extreme_inputs(self, output_dim):
        """Test that the layer handles extreme inputs without producing zeros or NaNs."""
        layer = RoutingProbabilitiesLayer(output_dim=output_dim)

        # Test with very large inputs
        large_inputs = tf.ones((4, 16)) * 100.0
        output_large = layer(large_inputs).numpy()
        assert np.all(output_large > 0), "Output contains zero probabilities with large inputs"
        assert not np.any(np.isnan(output_large)), "Output contains NaN with large inputs"

        # Test with very small inputs
        small_inputs = tf.ones((4, 16)) * -100.0
        output_small = layer(small_inputs).numpy()
        assert np.all(output_small > 0), "Output contains zero probabilities with small inputs"
        assert not np.any(np.isnan(output_small)), "Output contains NaN with small inputs"

        # Test with zero inputs
        zero_inputs = tf.zeros((4, 16))
        output_zero = layer(zero_inputs).numpy()
        assert np.all(output_zero > 0), "Output contains zero probabilities with zero inputs"
        assert not np.any(np.isnan(output_zero)), "Output contains NaN with zero inputs"

    def test_decision_weights_properties(self):
        """Test that decision weight patterns are properly constructed."""
        layer = RoutingProbabilitiesLayer(output_dim=10)
        inputs = tf.random.normal([1, 16])
        layer(inputs)  # Build layer

        # Check shape
        assert layer.decision_weights.shape == (layer.num_decisions, 16)

        # Check that weights are normalized (unit L2 norm for each decision)
        for i in range(layer.num_decisions):
            weights = layer.decision_weights[i, :]
            norm = tf.sqrt(tf.reduce_sum(tf.square(weights))).numpy()
            assert np.isclose(norm, 1.0, atol=1e-5), \
                f"Decision {i} weights are not normalized: norm={norm}"

    def test_serialization(self):
        """Test serialization and deserialization of the layer."""
        original_layer = RoutingProbabilitiesLayer(
            output_dim=12,
            epsilon=1e-6,
            name="test_routing"
        )
        # Build the layer
        inputs = tf.random.normal([1, 16])
        original_layer(inputs)

        config = original_layer.get_config()
        recreated_layer = RoutingProbabilitiesLayer.from_config(config)

        assert recreated_layer.name == original_layer.name
        assert recreated_layer.output_dim == original_layer.output_dim
        assert recreated_layer.epsilon == original_layer.epsilon

    def test_serialization_without_output_dim(self):
        """Test serialization when output_dim was inferred."""
        original_layer = RoutingProbabilitiesLayer(name="inferred_routing")
        # Build with inferred output_dim
        inputs = tf.random.normal([1, 10])
        original_output = original_layer(inputs)

        config = original_layer.get_config()
        recreated_layer = RoutingProbabilitiesLayer.from_config(config)

        # After building recreated layer, it should infer the same output_dim
        recreated_output = recreated_layer(inputs)
        assert recreated_layer.output_dim == 10

        # Outputs should match
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(original_output),
            keras.ops.convert_to_numpy(recreated_output),
            rtol=1e-6, atol=1e-6,
            err_msg="Recreated layer should produce same output"
        )

    @pytest.mark.parametrize("input_features, output_dim", [(16, 10), (64, 33)])
    def test_model_save_load(self, input_features, output_dim):
        """Test saving and loading a model with the HierarchicalRoutingBasicLayer."""
        inputs = keras.Input(shape=(input_features,))
        x = keras.layers.Dense(32, activation='relu')(inputs)
        outputs = RoutingProbabilitiesLayer(
            output_dim=output_dim,
            name="routing_output"
        )(x)
        model = keras.Model(inputs=inputs, outputs=outputs)

        input_data = tf.random.normal([4, input_features])
        original_prediction = model.predict(input_data, verbose=0)

        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, "model.keras")
            model.save(model_path)
            loaded_model = keras.models.load_model(model_path)
            loaded_prediction = loaded_model.predict(input_data, verbose=0)

            np.testing.assert_allclose(
                original_prediction,
                loaded_prediction,
                rtol=1e-6, atol=1e-6,
                err_msg="Loaded model should produce same predictions"
            )
            assert isinstance(
                loaded_model.get_layer("routing_output"),
                RoutingProbabilitiesLayer
            )

    def test_model_save_load_with_inferred_output_dim(self):
        """Test saving/loading a model where output_dim was inferred."""
        input_features = 16

        inputs = keras.Input(shape=(input_features,))
        logits = keras.layers.Dense(10)(inputs)
        # output_dim will be inferred as 10
        outputs = RoutingProbabilitiesLayer(name="routing_output")(logits)
        model = keras.Model(inputs=inputs, outputs=outputs)

        input_data = tf.random.normal([4, input_features])
        original_prediction = model.predict(input_data, verbose=0)

        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, "model.keras")
            model.save(model_path)
            loaded_model = keras.models.load_model(model_path)
            loaded_prediction = loaded_model.predict(input_data, verbose=0)

            np.testing.assert_allclose(
                original_prediction,
                loaded_prediction,
                rtol=1e-6, atol=1e-6,
                err_msg="Loaded model with inferred output_dim should match"
            )

    def test_gradient_flow_through_inputs(self):
        """Test that gradients flow through the layer to upstream layers."""
        # Create a simple model with trainable dense layer before routing
        inputs = keras.Input(shape=(16,))
        x = keras.layers.Dense(32, activation='relu', name='dense_layer')(inputs)
        outputs = RoutingProbabilitiesLayer(output_dim=10)(x)
        model = keras.Model(inputs=inputs, outputs=outputs)

        # The routing layer has no weights, but the dense layer does
        assert len(model.trainable_weights) == 2  # kernel and bias from dense

        input_data = tf.random.normal([4, 16])
        target_data = tf.one_hot(tf.constant([0, 1, 2, 3]), depth=10)

        with tf.GradientTape() as tape:
            predictions = model(input_data, training=True)
            loss = keras.losses.categorical_crossentropy(target_data, predictions)
            loss = tf.reduce_mean(loss)

        # Gradients should flow to the dense layer weights
        grads = tape.gradient(loss, model.trainable_weights)
        assert len(grads) == 2, "Should have gradients for dense layer weights"
        for grad in grads:
            assert grad is not None, "Gradient should not be None"
            assert not np.all(grad.numpy() == 0), "Gradients should not be all zero"

    def test_no_trainable_parameters(self):
        """Test that the layer itself has no trainable parameters."""
        layer = RoutingProbabilitiesLayer(output_dim=10)
        inputs = tf.random.normal([1, 16])
        layer(inputs)

        assert len(layer.trainable_weights) == 0, \
            "HierarchicalRoutingBasicLayer should have no trainable weights"
        assert len(layer.non_trainable_weights) == 0, \
            "HierarchicalRoutingBasicLayer should have no non-trainable weights"

    @pytest.mark.parametrize(
        "batch_size, input_dim, output_dim",
        [
            (16, 8, 5),  # Original case
            (32, 32, 17),  # Larger, non-power-of-2 case
            (8, 16, 2),  # Minimum output dimension
        ]
    )
    def test_in_training_loop(self, batch_size, input_dim, output_dim):
        """Test that the layer works in a training loop (training upstream layers)."""
        model = keras.Sequential([
            keras.Input(shape=(input_dim,)),
            keras.layers.Dense(16, activation='relu'),
            RoutingProbabilitiesLayer(output_dim=output_dim)
        ])
        model.compile(optimizer="adam", loss="categorical_crossentropy")

        x_train = tf.random.normal([batch_size, input_dim])
        y_train_indices = tf.random.uniform([batch_size], 0, output_dim, dtype=tf.int32)
        y_train = tf.one_hot(y_train_indices, depth=output_dim)

        initial_loss = model.evaluate(x_train, y_train, verbose=0)

        # Train for a few epochs
        history = model.fit(x_train, y_train, epochs=5, batch_size=4, verbose=0)

        final_loss = model.evaluate(x_train, y_train, verbose=0)

        # Loss should decrease (upstream Dense layer is learning)
        assert final_loss < initial_loss, \
            f"Loss should decrease: initial={initial_loss}, final={final_loss}"
        assert not np.isnan(final_loss), "Loss became NaN during training"

        # Check that all losses are finite
        assert all(np.isfinite(history.history['loss'])), \
            "Training losses should all be finite"

    def test_comparison_across_different_input_dimensions(self):
        """Test that the layer works with various input dimensions."""
        output_dim = 10

        for input_dim in [4, 8, 16, 32, 64, 128]:
            layer = RoutingProbabilitiesLayer(output_dim=output_dim)
            inputs = tf.random.normal([8, input_dim])
            output = layer(inputs)

            assert output.shape == (8, output_dim)
            sums = tf.reduce_sum(output, axis=1).numpy()
            assert np.allclose(sums, 1.0, atol=1e-6)

    def test_batch_independence(self):
        """Test that each sample in a batch is processed independently."""
        layer = RoutingProbabilitiesLayer(output_dim=10)

        # Define inputs
        input_1 = tf.constant([[1.0, 2.0, 3.0, 4.0]])
        input_2 = tf.constant([[5.0, 6.0, 7.0, 8.0]])

        # Process individually. The first call builds the layer.
        output_1_individual = layer(input_1)
        output_2_individual = layer(input_2)

        # Process as a batch (re-uses the same built layer)
        inputs_batch = tf.concat([input_1, input_2], axis=0)
        outputs_batch = layer(inputs_batch)

        # Results should match; relax tolerance for minor float discrepancies.
        # A tolerance of 3e-5 is reasonable for float32 backend variations.
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(output_1_individual),
            keras.ops.convert_to_numpy(outputs_batch[0:1, :]),
            rtol=1e-5, atol=3e-5,
            err_msg="First batch item should match individual processing"
        )
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(output_2_individual),
            keras.ops.convert_to_numpy(outputs_batch[1:2, :]),
            rtol=1e-5, atol=3e-5,
            err_msg="Second batch item should match individual processing"
        )

    def test_epsilon_parameter(self):
        """Test that epsilon parameter prevents numerical issues."""
        # Very small epsilon might cause issues
        layer_small_eps = RoutingProbabilitiesLayer(output_dim=10, epsilon=1e-10)
        layer_large_eps = RoutingProbabilitiesLayer(output_dim=10, epsilon=1e-5)

        inputs = tf.random.normal([8, 16])

        output_small = layer_small_eps(inputs).numpy()
        output_large = layer_large_eps(inputs).numpy()

        # Both should produce valid probabilities
        assert np.all(output_small > 0) and np.all(output_small <= 1.0)
        assert np.all(output_large > 0) and np.all(output_large <= 1.0)

        # Sums should be close to 1
        assert np.allclose(np.sum(output_small, axis=1), 1.0, atol=1e-6)
        assert np.allclose(np.sum(output_large, axis=1), 1.0, atol=1e-6)

    def test_as_softmax_replacement(self):
        """Test using the layer as a drop-in replacement for softmax."""
        input_features = 16
        output_dim = 10

        # Model with softmax
        inputs = keras.Input(shape=(input_features,))
        logits = keras.layers.Dense(output_dim)(inputs)
        outputs_softmax = keras.layers.Activation('softmax')(logits)
        model_softmax = keras.Model(inputs=inputs, outputs=outputs_softmax)

        # Model with hierarchical routing
        inputs2 = keras.Input(shape=(input_features,))
        logits2 = keras.layers.Dense(output_dim)(inputs2)
        outputs_routing = RoutingProbabilitiesLayer(output_dim=output_dim)(logits2)
        model_routing = keras.Model(inputs=inputs2, outputs=outputs_routing)

        # Both should work similarly in structure
        input_data = tf.random.normal([8, input_features])

        pred_softmax = model_softmax(input_data)
        pred_routing = model_routing(input_data)

        # Both should produce valid probability distributions
        assert pred_softmax.shape == pred_routing.shape
        assert np.allclose(np.sum(pred_softmax.numpy(), axis=1), 1.0, atol=1e-6)
        assert np.allclose(np.sum(pred_routing.numpy(), axis=1), 1.0, atol=1e-6)

    def test_inference_mode_matches_training_mode(self):
        """Test that training and inference modes produce identical results."""
        layer = RoutingProbabilitiesLayer(output_dim=10)
        inputs = tf.random.normal([8, 16])

        output_training = layer(inputs, training=True)
        output_inference = layer(inputs, training=False)

        # Since layer is deterministic and has no dropout/etc, outputs should match
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(output_training),
            keras.ops.convert_to_numpy(output_inference),
            rtol=1e-6, atol=1e-6,
            err_msg="Training and inference modes should produce identical outputs"
        )