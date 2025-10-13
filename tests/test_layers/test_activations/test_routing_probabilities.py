"""
Test suite for RoutingProbabilitiesLayer.

This module contains comprehensive tests for the parameter-free hierarchical
routing layer that acts as a drop-in replacement for softmax activation.
"""

import pytest
import numpy as np
import tensorflow as tf
import keras
import os
import tempfile

from dl_techniques.layers.activations.routing_probabilities import RoutingProbabilitiesLayer


# ==============================================================================
# Test Suite
# ==============================================================================


class TestRoutingProbabilitiesLayer:
    """
    Test suite for the non-trainable RoutingProbabilitiesLayer implementation.

    This suite tests the deterministic, parameter-free hierarchical routing layer
    that acts as a drop-in replacement for softmax activation with flexible axis support.
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
            self, output_dim: int, expected_padded: int, expected_decisions: int
    ) -> None:
        """Test initialization with explicit output_dim parameter."""
        layer = RoutingProbabilitiesLayer(output_dim=output_dim)
        assert isinstance(layer, keras.layers.Layer)
        assert layer.output_dim == output_dim
        assert layer.axis == -1  # Default axis
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

    @pytest.mark.parametrize("axis", [-1, -2, 0, 1, 2])
    def test_initialization_with_axis_parameter(self, axis: int) -> None:
        """Test initialization with different axis values."""
        layer = RoutingProbabilitiesLayer(output_dim=10, axis=axis)
        assert layer.axis == axis
        assert isinstance(layer, keras.layers.Layer)

    def test_initialization_without_output_dim(self) -> None:
        """Test initialization without output_dim (to be inferred)."""
        layer = RoutingProbabilitiesLayer()
        assert layer.output_dim is None
        assert layer.axis == -1  # Default
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

    def test_invalid_initialization(self) -> None:
        """Test that invalid parameters raise appropriate errors."""
        with pytest.raises(ValueError, match="must be an integer greater than 1"):
            RoutingProbabilitiesLayer(output_dim=1)
        with pytest.raises(ValueError, match="must be an integer greater than 1"):
            RoutingProbabilitiesLayer(output_dim=0)
        with pytest.raises(ValueError, match="must be an integer greater than 1"):
            RoutingProbabilitiesLayer(output_dim=-5)

    def test_invalid_axis_type(self) -> None:
        """Test that non-integer axis raises error."""
        with pytest.raises(ValueError, match="must be an integer"):
            RoutingProbabilitiesLayer(output_dim=10, axis=1.5)
        with pytest.raises(ValueError, match="must be an integer"):
            RoutingProbabilitiesLayer(output_dim=10, axis="last")

    def test_invalid_axis_out_of_bounds(self) -> None:
        """Test that out-of-bounds axis raises error during build."""
        layer = RoutingProbabilitiesLayer(output_dim=10, axis=5)
        inputs = tf.random.normal([2, 3, 4])  # 3D input

        with pytest.raises(ValueError, match="axis .* is out of bounds"):
            layer(inputs)

    def test_invalid_inference_none_input_dim(self) -> None:
        """Test that build fails when trying to infer from None dimension."""
        layer = RoutingProbabilitiesLayer()  # output_dim not provided

        with pytest.raises(ValueError, match="Cannot infer output_dim"):
            # Try to build with unknown last dimension
            layer.build((None, None))

    def test_invalid_inference_none_dimension_at_axis(self) -> None:
        """Test that build fails when axis dimension is None and output_dim not provided."""
        layer = RoutingProbabilitiesLayer(axis=1)  # output_dim not provided

        with pytest.raises(ValueError, match="Cannot infer output_dim"):
            # Try to build with None at axis 1
            layer.build((2, None, 4))

    @pytest.mark.parametrize("batch_size, input_features", [(1, 8), (4, 16), (32, 64)])
    def test_build_process(self, batch_size: int, input_features: int) -> None:
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
    def test_output_shapes_2d(
            self, batch_size: int, input_features: int, output_dim: int
    ) -> None:
        """Test that output shapes are computed correctly for 2D inputs."""
        layer = RoutingProbabilitiesLayer(output_dim=output_dim)
        inputs = tf.random.normal([batch_size, input_features])
        output = layer(inputs)

        expected_shape = (batch_size, output_dim)
        assert output.shape == expected_shape
        computed_shape = layer.compute_output_shape(inputs.shape)
        assert computed_shape == expected_shape

    @pytest.mark.parametrize(
        "input_shape, axis, output_dim, expected_shape",
        [
            ((2, 10), -1, 5, (2, 5)),  # 2D, last axis
            ((2, 10), 1, 5, (2, 5)),  # 2D, explicit last axis
            ((2, 3, 10), -1, 5, (2, 3, 5)),  # 3D, last axis
            ((2, 10, 3), 1, 5, (2, 5, 3)),  # 3D, middle axis
            ((2, 10, 3), -2, 5, (2, 5, 3)),  # 3D, middle axis (negative)
            ((4, 5, 6, 10), -1, 8, (4, 5, 6, 8)),  # 4D, last axis
            ((4, 5, 10, 6), 2, 8, (4, 5, 8, 6)),  # 4D, third axis
            ((4, 10, 5, 6), -3, 8, (4, 8, 5, 6)),  # 4D, second axis (negative)
        ]
    )
    def test_output_shapes_multidimensional(
            self,
            input_shape: tuple,
            axis: int,
            output_dim: int,
            expected_shape: tuple
    ) -> None:
        """Test output shapes for multi-dimensional inputs with various axes."""
        layer = RoutingProbabilitiesLayer(output_dim=output_dim, axis=axis)
        inputs = tf.random.normal(input_shape)
        output = layer(inputs)

        assert output.shape == expected_shape
        computed_shape = layer.compute_output_shape(inputs.shape)
        assert computed_shape == expected_shape

    @pytest.mark.parametrize(
        "batch_size, input_features, output_dim",
        [(1, 8, 5), (4, 16, 10), (32, 64, 128), (7, 21, 33)]
    )
    def test_output_is_normalized_2d(
            self, batch_size: int, input_features: int, output_dim: int
    ) -> None:
        """Test that the output is a valid probability distribution for 2D inputs."""
        layer = RoutingProbabilitiesLayer(output_dim=output_dim)
        inputs = tf.random.normal([batch_size, input_features])
        output = layer(inputs)

        sums = tf.reduce_sum(output, axis=-1).numpy()
        assert np.allclose(sums, 1.0, atol=1e-6), \
            f"Probabilities don't sum to 1: {sums}"

    @pytest.mark.parametrize(
        "input_shape, axis",
        [
            ((4, 10), -1),
            ((4, 10), 1),
            ((2, 3, 10), -1),
            ((2, 10, 3), 1),
            ((2, 10, 3), -2),
            ((4, 5, 6, 10), 2),
        ]
    )
    def test_output_is_normalized_multidimensional(
            self, input_shape: tuple, axis: int
    ) -> None:
        """Test normalization for multi-dimensional inputs along specified axis."""
        layer = RoutingProbabilitiesLayer(axis=axis)
        inputs = tf.random.normal(input_shape)
        output = layer(inputs)

        # Normalize axis for reduction
        normalized_axis = axis if axis >= 0 else len(input_shape) + axis
        sums = tf.reduce_sum(output, axis=normalized_axis).numpy()

        assert np.allclose(sums, 1.0, atol=1e-6), \
            f"Probabilities don't sum to 1 along axis {axis}: {sums}"

    @pytest.mark.parametrize("output_dim", [2, 5, 10, 32, 100])
    def test_output_values_in_valid_range(self, output_dim: int) -> None:
        """Test that all output probabilities are in [0, 1] range."""
        layer = RoutingProbabilitiesLayer(output_dim=output_dim)
        inputs = tf.random.normal([16, 32])
        output = layer(inputs).numpy()

        assert np.all(output >= 0.0), "Found negative probabilities"
        assert np.all(output <= 1.0), "Found probabilities > 1"
        assert np.all(output > 0.0), "Found zero probabilities (should be > epsilon)"

    def test_deterministic_behavior(self) -> None:
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

    def test_different_inputs_produce_different_outputs(self) -> None:
        """Test that different inputs produce different probability distributions."""
        layer = RoutingProbabilitiesLayer(output_dim=10)

        inputs_1 = tf.constant([[1.0, 2.0, 3.0, 4.0, 5.0]])
        inputs_2 = tf.constant([[5.0, 4.0, 3.0, 2.0, 1.0]])

        output_1 = layer(inputs_1).numpy()
        output_2 = layer(inputs_2).numpy()

        # Outputs should be different
        assert not np.allclose(output_1, output_2, atol=1e-3), \
            "Different inputs should produce different outputs"

    def test_power_of_2_output_dim(self) -> None:
        """Test handling of power-of-2 output dimensions (no renormalization)."""
        for output_dim in [2, 4, 8, 16, 32]:
            layer = RoutingProbabilitiesLayer(output_dim=output_dim)
            inputs = tf.random.normal([8, 16])
            output = layer(inputs)

            assert output.shape == (8, output_dim)
            # Should still be normalized
            sums = tf.reduce_sum(output, axis=-1).numpy()
            assert np.allclose(sums, 1.0, atol=1e-6)

    def test_non_power_of_2_output_dim(self) -> None:
        """Test handling of non-power-of-2 output dimensions (with renormalization)."""
        for output_dim in [3, 5, 7, 13, 17, 100]:
            layer = RoutingProbabilitiesLayer(output_dim=output_dim)
            inputs = tf.random.normal([8, 16])
            output = layer(inputs)

            assert output.shape == (8, output_dim)
            # Should be normalized after slicing and renormalization
            sums = tf.reduce_sum(output, axis=-1).numpy()
            assert np.allclose(sums, 1.0, atol=1e-6)

    @pytest.mark.parametrize("output_dim", [2, 10, 32, 100])
    def test_stability_with_extreme_inputs(self, output_dim: int) -> None:
        """Test that the layer handles extreme inputs without producing zeros or NaNs."""
        layer = RoutingProbabilitiesLayer(output_dim=output_dim)

        # Test with very large inputs
        large_inputs = tf.ones((4, 16)) * 100.0
        output_large = layer(large_inputs).numpy()
        assert np.all(output_large > 0), \
            "Output contains zero probabilities with large inputs"
        assert not np.any(np.isnan(output_large)), \
            "Output contains NaN with large inputs"

        # Test with very small inputs
        small_inputs = tf.ones((4, 16)) * -100.0
        output_small = layer(small_inputs).numpy()
        assert np.all(output_small > 0), \
            "Output contains zero probabilities with small inputs"
        assert not np.any(np.isnan(output_small)), \
            "Output contains NaN with small inputs"

        # Test with zero inputs
        zero_inputs = tf.zeros((4, 16))
        output_zero = layer(zero_inputs).numpy()
        assert np.all(output_zero > 0), \
            "Output contains zero probabilities with zero inputs"
        assert not np.any(np.isnan(output_zero)), \
            "Output contains NaN with zero inputs"

    def test_decision_weights_properties(self) -> None:
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

    def test_serialization(self) -> None:
        """Test serialization and deserialization of the layer."""
        original_layer = RoutingProbabilitiesLayer(
            output_dim=12,
            axis=-1,
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
        assert recreated_layer.axis == original_layer.axis
        assert recreated_layer.epsilon == original_layer.epsilon

    def test_serialization_with_custom_axis(self) -> None:
        """Test serialization with non-default axis."""
        original_layer = RoutingProbabilitiesLayer(
            output_dim=10,
            axis=1,
            name="routing_axis1"
        )
        # Build with 3D input
        inputs = tf.random.normal([2, 15, 4])
        original_output = original_layer(inputs)

        config = original_layer.get_config()
        recreated_layer = RoutingProbabilitiesLayer.from_config(config)

        recreated_output = recreated_layer(inputs)
        assert recreated_layer.axis == 1

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(original_output),
            keras.ops.convert_to_numpy(recreated_output),
            rtol=1e-6, atol=1e-6,
            err_msg="Recreated layer should produce same output"
        )

    def test_serialization_without_output_dim(self) -> None:
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
    def test_model_save_load(self, input_features: int, output_dim: int) -> None:
        """Test saving and loading a model with the RoutingProbabilitiesLayer."""
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

    def test_model_save_load_with_inferred_output_dim(self) -> None:
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

    def test_model_save_load_with_custom_axis(self) -> None:
        """Test saving/loading model with non-default axis."""
        inputs = keras.Input(shape=(8, 15))
        outputs = RoutingProbabilitiesLayer(
            output_dim=10,
            axis=1,
            name="routing_output"
        )(inputs)
        model = keras.Model(inputs=inputs, outputs=outputs)

        input_data = tf.random.normal([2, 8, 15])
        original_prediction = model.predict(input_data, verbose=0)

        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, "model.keras")
            model.save(model_path)
            loaded_model = keras.models.load_model(model_path)
            loaded_prediction = loaded_model.predict(input_data, verbose=0)

            # axis=1 modifies dimension at index 1: (2, 8, 15) -> (2, 10, 15)
            assert loaded_prediction.shape == (2, 10, 15)
            np.testing.assert_allclose(
                original_prediction,
                loaded_prediction,
                rtol=1e-6, atol=1e-6,
                err_msg="Loaded model with custom axis should match"
            )

    def test_gradient_flow_through_inputs(self) -> None:
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
            assert not np.all(grad.numpy() == 0), \
                "Gradients should not be all zero"

    def test_no_trainable_parameters(self) -> None:
        """Test that the layer itself has no trainable parameters."""
        layer = RoutingProbabilitiesLayer(output_dim=10)
        inputs = tf.random.normal([1, 16])
        layer(inputs)

        assert len(layer.trainable_weights) == 0, \
            "RoutingProbabilitiesLayer should have no trainable weights"
        assert len(layer.non_trainable_weights) == 0, \
            "RoutingProbabilitiesLayer should have no non-trainable weights"

    @pytest.mark.parametrize(
        "batch_size, input_dim, output_dim",
        [
            (16, 8, 5),  # Original case
            (32, 32, 17),  # Larger, non-power-of-2 case
            (8, 16, 2),  # Minimum output dimension
        ]
    )
    def test_in_training_loop(
            self, batch_size: int, input_dim: int, output_dim: int
    ) -> None:
        """Test that the layer works in a training loop (training upstream layers)."""
        model = keras.Sequential([
            keras.Input(shape=(input_dim,)),
            keras.layers.Dense(16, activation='relu'),
            RoutingProbabilitiesLayer(output_dim=output_dim)
        ])
        model.compile(optimizer="adam", loss="categorical_crossentropy")

        x_train = tf.random.normal([batch_size, input_dim])
        y_train_indices = tf.random.uniform(
            [batch_size], 0, output_dim, dtype=tf.int32
        )
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

    def test_comparison_across_different_input_dimensions(self) -> None:
        """Test that the layer works with various input dimensions."""
        output_dim = 10

        for input_dim in [4, 8, 16, 32, 64, 128]:
            layer = RoutingProbabilitiesLayer(output_dim=output_dim)
            inputs = tf.random.normal([8, input_dim])
            output = layer(inputs)

            assert output.shape == (8, output_dim)
            sums = tf.reduce_sum(output, axis=-1).numpy()
            assert np.allclose(sums, 1.0, atol=1e-6)

    def test_batch_independence(self) -> None:
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

        # Results should match
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

    def test_epsilon_parameter(self) -> None:
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
        assert np.allclose(np.sum(output_small, axis=-1), 1.0, atol=1e-6)
        assert np.allclose(np.sum(output_large, axis=-1), 1.0, atol=1e-6)

    def test_as_softmax_replacement(self) -> None:
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
        assert np.allclose(np.sum(pred_softmax.numpy(), axis=-1), 1.0, atol=1e-6)
        assert np.allclose(np.sum(pred_routing.numpy(), axis=-1), 1.0, atol=1e-6)

    def test_inference_mode_matches_training_mode(self) -> None:
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

    # -------------------------------------------------------------------------
    # Tests for multi-dimensional inputs and axis functionality
    # -------------------------------------------------------------------------

    @pytest.mark.parametrize(
        "input_shape, axis",
        [
            ((4, 10), -1),  # 2D, last axis
            ((4, 10), 1),   # 2D, explicit axis
            ((2, 3, 10), -1),  # 3D, last axis
            ((2, 10, 3), 1),   # 3D, middle axis
            ((2, 10, 3), -2),  # 3D, middle axis (negative)
            ((4, 5, 6, 10), -1),  # 4D, last axis
            ((4, 5, 10, 6), 2),   # 4D, third axis
            ((4, 10, 5, 6), 1),   # 4D, second axis
        ]
    )
    def test_axis_functionality(self, input_shape: tuple, axis: int) -> None:
        """Test that axis parameter works correctly for various shapes."""
        layer = RoutingProbabilitiesLayer(axis=axis)
        inputs = tf.random.normal(input_shape)
        output = layer(inputs)

        # Output shape should match input except at the specified axis
        assert output.shape == input_shape

        # Verify normalization along the correct axis
        normalized_axis = axis if axis >= 0 else len(input_shape) + axis
        sums = tf.reduce_sum(output, axis=normalized_axis).numpy()
        assert np.allclose(sums, 1.0, atol=1e-6), \
            f"Not normalized along axis {axis}"

    def test_negative_axis_normalization(self) -> None:
        """Test that negative axis values are correctly normalized."""
        # Create layer with negative axis
        layer = RoutingProbabilitiesLayer(output_dim=8, axis=-2)

        # Build with 3D input: (batch, target_axis, other)
        inputs = tf.random.normal([2, 10, 5])
        layer(inputs)

        # Check that normalized axis is correct (axis 1 for shape (2, 10, 5))
        assert layer._normalized_axis == 1

    @pytest.mark.parametrize(
        "input_shape, axis, output_dim",
        [
            ((4, 15), 1, 10),        # 2D
            ((2, 3, 15), 2, 10),     # 3D, last axis
            ((2, 15, 3), 1, 10),     # 3D, middle axis
            ((4, 5, 15, 6), 2, 10),  # 4D, third axis
        ]
    )
    def test_axis_with_explicit_output_dim(
            self,
            input_shape: tuple,
            axis: int,
            output_dim: int
    ) -> None:
        """Test axis functionality with explicit output_dim."""
        layer = RoutingProbabilitiesLayer(output_dim=output_dim, axis=axis)
        inputs = tf.random.normal(input_shape)
        output = layer(inputs)

        # Build expected output shape
        expected_shape = list(input_shape)
        normalized_axis = axis if axis >= 0 else len(input_shape) + axis
        expected_shape[normalized_axis] = output_dim

        assert output.shape == tuple(expected_shape)

        # Verify normalization
        sums = tf.reduce_sum(output, axis=normalized_axis).numpy()
        assert np.allclose(sums, 1.0, atol=1e-6)

    def test_3d_input_axis_last(self) -> None:
        """Test 3D input with routing on last axis."""
        batch, seq_len, features = 4, 8, 10
        layer = RoutingProbabilitiesLayer(output_dim=5, axis=-1)

        inputs = tf.random.normal([batch, seq_len, features])
        output = layer(inputs)

        assert output.shape == (batch, seq_len, 5)

        # Check normalization for each position in sequence
        for i in range(seq_len):
            sums = tf.reduce_sum(output[:, i, :], axis=-1).numpy()
            assert np.allclose(sums, 1.0, atol=1e-6)

    def test_3d_input_axis_middle(self) -> None:
        """Test 3D input with routing on middle axis."""
        batch, classes, features = 4, 10, 8
        layer = RoutingProbabilitiesLayer(output_dim=5, axis=1)

        inputs = tf.random.normal([batch, classes, features])
        output = layer(inputs)

        assert output.shape == (batch, 5, features)

        # Check normalization along axis 1
        sums = tf.reduce_sum(output, axis=1).numpy()
        assert np.allclose(sums, 1.0, atol=1e-6)

    def test_4d_input_various_axes(self) -> None:
        """Test 4D input with routing on various axes."""
        shape = (2, 3, 4, 10)

        for axis in range(-4, 4):
            layer = RoutingProbabilitiesLayer(output_dim=5, axis=axis)
            inputs = tf.random.normal(shape)
            output = layer(inputs)

            # Calculate expected shape
            expected_shape = list(shape)
            normalized_axis = axis if axis >= 0 else len(shape) + axis
            expected_shape[normalized_axis] = 5

            assert output.shape == tuple(expected_shape), \
                f"Failed for axis {axis}"

            # Verify normalization
            sums = tf.reduce_sum(output, axis=normalized_axis).numpy()
            assert np.allclose(sums, 1.0, atol=1e-6), \
                f"Not normalized for axis {axis}"

    def test_axis_consistency_across_calls(self) -> None:
        """Test that the same axis produces consistent results."""
        layer = RoutingProbabilitiesLayer(output_dim=8, axis=1)
        inputs = tf.constant([
            [[1.0, 2.0, 3.0, 4.0, 5.0] for _ in range(10)]
            for _ in range(2)
        ])

        output_1 = layer(inputs)
        output_2 = layer(inputs)

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(output_1),
            keras.ops.convert_to_numpy(output_2),
            rtol=1e-6, atol=1e-6,
            err_msg="Outputs should be consistent across calls"
        )

    def test_different_axes_produce_different_outputs(self) -> None:
        """Test that using different axes produces different outputs."""
        inputs = tf.random.normal([2, 10, 10])

        layer_axis_1 = RoutingProbabilitiesLayer(output_dim=8, axis=1)
        layer_axis_2 = RoutingProbabilitiesLayer(output_dim=8, axis=2)

        output_axis_1 = layer_axis_1(inputs)  # Shape: (2, 8, 10)
        output_axis_2 = layer_axis_2(inputs)  # Shape: (2, 10, 8)

        # Shapes should be different
        assert output_axis_1.shape != output_axis_2.shape
        assert output_axis_1.shape == (2, 8, 10)
        assert output_axis_2.shape == (2, 10, 8)

    def test_sequence_classification_use_case(self) -> None:
        """Test typical sequence classification use case."""
        # Simulate: (batch, sequence_length, features) -> (batch, sequence_length, classes)
        batch_size, seq_len, feature_dim = 8, 16, 32
        num_classes = 10

        inputs = keras.Input(shape=(seq_len, feature_dim))
        x = keras.layers.Dense(64, activation='relu')(inputs)
        # Apply routing to get probabilities for each position in sequence
        outputs = RoutingProbabilitiesLayer(output_dim=num_classes, axis=-1)(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        test_input = tf.random.normal([batch_size, seq_len, feature_dim])
        predictions = model(test_input)

        assert predictions.shape == (batch_size, seq_len, num_classes)

        # Each position should have valid probability distribution
        for b in range(batch_size):
            for s in range(seq_len):
                probs = predictions[b, s, :].numpy()
                assert np.allclose(np.sum(probs), 1.0, atol=1e-6)
                assert np.all(probs > 0) and np.all(probs <= 1.0)

    def test_image_feature_map_use_case(self) -> None:
        """Test using routing on image feature maps."""
        # Simulate: (batch, height, width, channels) -> (batch, height, width, classes)
        batch, h, w, channels = 4, 8, 8, 64
        num_classes = 10

        inputs = keras.Input(shape=(h, w, channels))
        # Apply routing along channel dimension
        outputs = RoutingProbabilitiesLayer(output_dim=num_classes, axis=-1)(inputs)

        model = keras.Model(inputs=inputs, outputs=outputs)

        test_input = tf.random.normal([batch, h, w, channels])
        predictions = model(test_input)

        assert predictions.shape == (batch, h, w, num_classes)

        # Each spatial location should have valid probability distribution
        sums = tf.reduce_sum(predictions, axis=-1).numpy()
        assert np.allclose(sums, 1.0, atol=1e-6)