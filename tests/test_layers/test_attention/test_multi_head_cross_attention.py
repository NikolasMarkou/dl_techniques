"""Test suite for MultiHeadCrossAttention and RoutingProbabilitiesLayer.

This module contains comprehensive tests for both the MultiHeadCrossAttention layer
and the RoutingProbabilitiesLayer, validating their functionality independently and
in integration, including cross-attention, self-attention, masking, adaptive softmax,
hierarchical routing, and serialization.
"""

import pytest
import numpy as np
import tensorflow as tf
import keras
import tempfile
import os

# Import the layers to test
from dl_techniques.layers.attention.multi_head_cross_attention import MultiHeadCrossAttention
from dl_techniques.layers.activations.routing_probabilities import RoutingProbabilitiesLayer


class TestRoutingProbabilitiesLayer:
    """Test suite for RoutingProbabilitiesLayer."""

    # ==================== Initialization Tests ====================

    def test_initialization_with_output_dim(self):
        """Test initialization with explicit output_dim."""
        layer = RoutingProbabilitiesLayer(output_dim=10)
        assert layer.output_dim == 10
        assert layer.axis == -1
        assert layer.epsilon == 1e-7

    def test_initialization_without_output_dim(self):
        """Test initialization without output_dim (will be inferred)."""
        layer = RoutingProbabilitiesLayer()
        assert layer.output_dim is None
        assert layer.axis == -1

    def test_initialization_custom_axis(self):
        """Test initialization with custom axis."""
        layer = RoutingProbabilitiesLayer(output_dim=10, axis=1)
        assert layer.axis == 1

        layer_neg = RoutingProbabilitiesLayer(output_dim=10, axis=-2)
        assert layer_neg.axis == -2

    def test_initialization_custom_epsilon(self):
        """Test initialization with custom epsilon."""
        layer = RoutingProbabilitiesLayer(output_dim=10, epsilon=1e-5)
        assert layer.epsilon == 1e-5

    def test_invalid_output_dim(self):
        """Test that invalid output_dim raises ValueError."""
        with pytest.raises(ValueError, match="must be an integer greater than 1"):
            RoutingProbabilitiesLayer(output_dim=1)

        with pytest.raises(ValueError, match="must be an integer greater than 1"):
            RoutingProbabilitiesLayer(output_dim=0)

        with pytest.raises(ValueError, match="must be an integer greater than 1"):
            RoutingProbabilitiesLayer(output_dim=-5)

    def test_invalid_axis_type(self):
        """Test that non-integer axis raises ValueError."""
        with pytest.raises(ValueError, match="must be an integer"):
            RoutingProbabilitiesLayer(output_dim=10, axis=1.5)

    # ==================== Build Process Tests ====================

    def test_build_with_explicit_output_dim(self):
        """Test build process with explicit output_dim."""
        layer = RoutingProbabilitiesLayer(output_dim=10)
        layer.build((None, 20, 15))

        assert layer.built is True
        assert layer.padded_output_dim == 16  # Next power of 2 >= 10
        assert layer.num_decisions == 4  # log2(16)
        assert layer.decision_weights.shape == (4, 15)  # (num_decisions, input_dim)

    def test_build_with_inferred_output_dim(self):
        """Test build process with output_dim inferred from input."""
        layer = RoutingProbabilitiesLayer()
        layer.build((None, 20, 10))

        assert layer.output_dim == 10
        assert layer.padded_output_dim == 16
        assert layer.num_decisions == 4

    def test_build_with_power_of_two_output(self):
        """Test build with output_dim that is already a power of 2."""
        layer = RoutingProbabilitiesLayer(output_dim=8)
        layer.build((None, 10, 20))

        assert layer.padded_output_dim == 8
        assert layer.num_decisions == 3

    def test_build_axis_normalization(self):
        """Test that negative axis is normalized correctly."""
        layer = RoutingProbabilitiesLayer(output_dim=10, axis=-1)
        layer.build((None, 20, 15))
        assert layer._normalized_axis == 2

        layer2 = RoutingProbabilitiesLayer(output_dim=10, axis=-2)
        layer2.build((None, 20, 15))
        assert layer2._normalized_axis == 1

    def test_build_fails_with_none_dimension_and_no_output_dim(self):
        """Test that build fails when axis dimension is None and output_dim not provided."""
        layer = RoutingProbabilitiesLayer()
        with pytest.raises(ValueError, match="Cannot infer output_dim"):
            layer.build((None, 20, None))

    def test_build_fails_with_invalid_axis(self):
        """Test that build fails with out-of-bounds axis."""
        layer = RoutingProbabilitiesLayer(output_dim=10, axis=5)
        with pytest.raises(ValueError, match="axis .* is out of bounds"):
            layer.build((None, 20, 15))

    def test_decision_weights_properties(self):
        """Test properties of generated decision weights."""
        layer = RoutingProbabilitiesLayer(output_dim=10)
        layer.build((None, 20))

        # Check shape
        assert layer.decision_weights.shape[0] == layer.num_decisions
        assert layer.decision_weights.shape[1] == 20

        # Check that weights are normalized (unit L2 norm)
        for i in range(layer.num_decisions):
            weights = layer.decision_weights[i]
            norm = tf.sqrt(tf.reduce_sum(tf.square(weights)))
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(norm),
                1.0,
                rtol=1e-5, atol=1e-5,
                err_msg=f"Decision weights {i} should have unit norm"
            )

    # ==================== Output Shape Tests ====================

    def test_compute_output_shape_2d(self):
        """Test output shape computation for 2D inputs."""
        layer = RoutingProbabilitiesLayer(output_dim=10)
        input_shape = (None, 20)
        output_shape = layer.compute_output_shape(input_shape)
        assert output_shape == (None, 10)

    def test_compute_output_shape_3d(self):
        """Test output shape computation for 3D inputs."""
        layer = RoutingProbabilitiesLayer(output_dim=5)
        input_shape = (None, 32, 20)
        output_shape = layer.compute_output_shape(input_shape)
        assert output_shape == (None, 32, 5)

    def test_compute_output_shape_different_axis(self):
        """Test output shape computation with different axis."""
        layer = RoutingProbabilitiesLayer(output_dim=8, axis=1)
        input_shape = (None, 20, 15)
        output_shape = layer.compute_output_shape(input_shape)
        assert output_shape == (None, 8, 15)

    def test_compute_output_shape_inferred_output_dim(self):
        """Test output shape when output_dim will be inferred."""
        layer = RoutingProbabilitiesLayer()
        input_shape = (None, 32, 10)
        output_shape = layer.compute_output_shape(input_shape)
        # When output_dim is None before build, shape should be preserved
        assert output_shape == input_shape

    # ==================== Forward Pass Tests ====================

    def test_forward_pass_2d_input(self):
        """Test forward pass with 2D input."""
        layer = RoutingProbabilitiesLayer(output_dim=10)
        inputs = tf.random.normal([4, 20])
        outputs = layer(inputs)

        assert outputs.shape == (4, 10)
        assert not tf.reduce_any(tf.math.is_nan(outputs))
        assert not tf.reduce_any(tf.math.is_inf(outputs))

    def test_forward_pass_3d_input(self):
        """Test forward pass with 3D input."""
        layer = RoutingProbabilitiesLayer(output_dim=5)
        inputs = tf.random.normal([2, 8, 16])
        outputs = layer(inputs)

        assert outputs.shape == (2, 8, 5)
        assert not tf.reduce_any(tf.math.is_nan(outputs))

    def test_forward_pass_4d_input(self):
        """Test forward pass with 4D input (batch, height, width, channels)."""
        layer = RoutingProbabilitiesLayer(output_dim=10, axis=-1)
        inputs = tf.random.normal([2, 8, 8, 20])
        outputs = layer(inputs)

        assert outputs.shape == (2, 8, 8, 10)
        assert not tf.reduce_any(tf.math.is_nan(outputs))

    def test_forward_pass_different_axis(self):
        """Test forward pass with routing on different axes."""
        inputs = tf.random.normal([2, 16, 10])

        # Apply on last axis (default)
        layer_last = RoutingProbabilitiesLayer(output_dim=5, axis=-1)
        output_last = layer_last(inputs)
        assert output_last.shape == (2, 16, 5)

        # Apply on middle axis
        layer_mid = RoutingProbabilitiesLayer(output_dim=8, axis=1)
        output_mid = layer_mid(inputs)
        assert output_mid.shape == (2, 8, 10)

    def test_probability_distribution_valid(self):
        """Test that output forms a valid probability distribution."""
        layer = RoutingProbabilitiesLayer(output_dim=10)
        inputs = tf.random.normal([5, 20])
        outputs = layer(inputs)

        # Check all probabilities are in [0, 1]
        assert tf.reduce_all(outputs >= 0.0)
        assert tf.reduce_all(outputs <= 1.0)

        # Check probabilities sum to 1.0 along the routing axis
        prob_sums = tf.reduce_sum(outputs, axis=-1)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(prob_sums),
            np.ones(5),
            rtol=1e-5, atol=1e-5,
            err_msg="Probabilities should sum to 1.0"
        )

    def test_probability_distribution_power_of_two(self):
        """Test probability distribution when output_dim is power of 2."""
        layer = RoutingProbabilitiesLayer(output_dim=8)
        inputs = tf.random.normal([3, 16])
        outputs = layer(inputs)

        prob_sums = tf.reduce_sum(outputs, axis=-1)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(prob_sums),
            np.ones(3),
            rtol=1e-5, atol=1e-5,
            err_msg="Probabilities should sum to 1.0 for power-of-2 dimensions"
        )

    def test_deterministic_output(self):
        """Test that layer produces deterministic output (no randomness)."""
        layer = RoutingProbabilitiesLayer(output_dim=10)
        inputs = tf.random.normal([2, 20])

        output1 = layer(inputs, training=True)
        output2 = layer(inputs, training=True)
        output3 = layer(inputs, training=False)

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(output1),
            keras.ops.convert_to_numpy(output2),
            rtol=1e-6, atol=1e-6,
            err_msg="Output should be deterministic"
        )

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(output1),
            keras.ops.convert_to_numpy(output3),
            rtol=1e-6, atol=1e-6,
            err_msg="Output should be same in training and inference"
        )

    # ==================== Edge Case Tests ====================

    def test_single_batch(self):
        """Test with single batch item."""
        layer = RoutingProbabilitiesLayer(output_dim=7)
        inputs = tf.random.normal([1, 15])
        outputs = layer(inputs)

        assert outputs.shape == (1, 7)
        prob_sum = tf.reduce_sum(outputs)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(prob_sum),
            1.0,
            rtol=1e-5, atol=1e-5
        )

    def test_two_class_output(self):
        """Test with minimum valid output_dim (2 classes)."""
        layer = RoutingProbabilitiesLayer(output_dim=2)
        inputs = tf.random.normal([4, 10])
        outputs = layer(inputs)

        assert outputs.shape == (4, 2)
        assert layer.padded_output_dim == 2
        assert layer.num_decisions == 1

    def test_large_output_dim(self):
        """Test with large output dimension."""
        layer = RoutingProbabilitiesLayer(output_dim=100)
        inputs = tf.random.normal([2, 50])
        outputs = layer(inputs)

        assert outputs.shape == (2, 100)
        assert layer.padded_output_dim == 128  # Next power of 2
        prob_sums = tf.reduce_sum(outputs, axis=-1)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(prob_sums),
            np.ones(2),
            rtol=1e-5, atol=1e-5
        )

    def test_numerical_stability_extreme_inputs(self):
        """Test numerical stability with extreme input values."""
        layer = RoutingProbabilitiesLayer(output_dim=10)

        test_cases = [
            tf.zeros((2, 20)),
            tf.ones((2, 20)) * 1e-10,
            tf.ones((2, 20)) * 1e3,
            tf.random.normal((2, 20)) * 100,
            tf.random.normal((2, 20)) * 0.01,
        ]

        for i, inputs in enumerate(test_cases):
            outputs = layer(inputs)
            assert not tf.reduce_any(tf.math.is_nan(outputs)), f"NaN in test case {i}"
            assert not tf.reduce_any(tf.math.is_inf(outputs)), f"Inf in test case {i}"
            assert tf.reduce_all(outputs >= 0.0), f"Negative probs in test case {i}"
            assert tf.reduce_all(outputs <= 1.0), f"Probs > 1.0 in test case {i}"

    def test_zero_inputs(self):
        """Test with all-zero inputs."""
        layer = RoutingProbabilitiesLayer(output_dim=5)
        inputs = tf.zeros((3, 10))
        outputs = layer(inputs)

        # With zero inputs, all logits should be zero, leading to sigmoid(0) = 0.5
        # This should still produce valid probabilities
        assert not tf.reduce_any(tf.math.is_nan(outputs))
        prob_sums = tf.reduce_sum(outputs, axis=-1)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(prob_sums),
            np.ones(3),
            rtol=1e-5, atol=1e-5
        )

    # ==================== Serialization Tests ====================

    def test_get_config(self):
        """Test that get_config captures all parameters."""
        layer = RoutingProbabilitiesLayer(output_dim=15, axis=-2, epsilon=1e-6)
        config = layer.get_config()

        assert "output_dim" in config
        assert "axis" in config
        assert "epsilon" in config
        assert config["output_dim"] == 15
        assert config["axis"] == -2
        assert config["epsilon"] == 1e-6

    def test_from_config(self):
        """Test recreating layer from config."""
        original_layer = RoutingProbabilitiesLayer(output_dim=12, axis=1, epsilon=1e-5)
        config = original_layer.get_config()

        recreated_layer = RoutingProbabilitiesLayer.from_config(config)
        assert recreated_layer.output_dim == 12
        assert recreated_layer.axis == 1
        assert recreated_layer.epsilon == 1e-5

    def test_serialization_roundtrip(self):
        """Test full serialization roundtrip."""
        layer = RoutingProbabilitiesLayer(output_dim=8)
        inputs = tf.random.normal([3, 20])
        layer.build(inputs.shape)

        original_output = layer(inputs)

        # Serialize and deserialize
        config = layer.get_config()
        new_layer = RoutingProbabilitiesLayer.from_config(config)
        new_layer.build(inputs.shape)

        new_output = new_layer(inputs)

        # Outputs should match (deterministic layer)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(original_output),
            keras.ops.convert_to_numpy(new_output),
            rtol=1e-6, atol=1e-6,
            err_msg="Outputs should match after serialization roundtrip"
        )

    # ==================== Model Integration Tests ====================

    def test_model_integration_as_activation(self):
        """Test using routing layer as an alternative to softmax."""
        inputs = keras.Input(shape=(32,))
        x = keras.layers.Dense(64)(inputs)
        x = keras.layers.Dense(10)(x)  # Logits
        outputs = RoutingProbabilitiesLayer(output_dim=10)(x)

        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer="adam", loss="categorical_crossentropy")

        # Test forward pass
        test_inputs = tf.random.normal([5, 32])
        predictions = model(test_inputs)

        assert predictions.shape == (5, 10)
        assert not tf.reduce_any(tf.math.is_nan(predictions))

    def test_model_integration_multilabel(self):
        """Test in a multi-label classification scenario."""
        inputs = keras.Input(shape=(16, 20))
        x = keras.layers.Dense(10)(inputs)
        outputs = RoutingProbabilitiesLayer(axis=-1)(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        test_inputs = tf.random.normal([2, 16, 20])
        predictions = model(test_inputs)

        assert predictions.shape == (2, 16, 10)

    def test_model_save_load(self):
        """Test saving and loading a model with routing layer."""
        inputs = keras.Input(shape=(20,))
        x = keras.layers.Dense(15)(inputs)
        outputs = RoutingProbabilitiesLayer(output_dim=10, name="routing")(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        test_inputs = tf.random.normal([3, 20])
        original_prediction = model.predict(test_inputs, verbose=0)

        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, "model.keras")
            model.save(model_path)

            loaded_model = keras.models.load_model(
                model_path,
                custom_objects={"RoutingProbabilitiesLayer": RoutingProbabilitiesLayer}
            )

            loaded_prediction = loaded_model.predict(test_inputs, verbose=0)

            np.testing.assert_allclose(
                original_prediction, loaded_prediction,
                rtol=1e-5, atol=1e-5,
                err_msg="Predictions should match after save/load"
            )

            assert isinstance(loaded_model.get_layer("routing"), RoutingProbabilitiesLayer)

    # ==================== Gradient Tests ====================

    def test_gradient_flow(self):
        """Test that gradients flow correctly through the layer."""
        layer = RoutingProbabilitiesLayer(output_dim=10)

        with tf.GradientTape() as tape:
            inputs = tf.Variable(tf.random.normal([4, 20]))
            outputs = layer(inputs)
            loss = tf.reduce_mean(outputs)

        gradients = tape.gradient(loss, inputs)

        assert gradients is not None, "Gradients should not be None"
        assert not tf.reduce_any(tf.math.is_nan(gradients)), "Gradients should not contain NaN"

    def test_trainable_false(self):
        """Test that layer has no trainable parameters."""
        layer = RoutingProbabilitiesLayer(output_dim=10)
        layer.build((None, 20))

        assert len(layer.trainable_variables) == 0, "Layer should have no trainable variables"
        assert len(layer.non_trainable_variables) == 0, "Layer should have no non-trainable variables"

    # ==================== Comparison Tests ====================

    def test_different_from_softmax(self):
        """Test that routing produces different output than softmax."""
        inputs = tf.random.normal([5, 20])

        # Routing layer
        routing_layer = RoutingProbabilitiesLayer(output_dim=10)
        routing_output = routing_layer(inputs)

        # Softmax on projected inputs
        dense = keras.layers.Dense(10)
        logits = dense(inputs)
        softmax_output = keras.ops.softmax(logits, axis=-1)

        # Outputs should be different (routing uses deterministic patterns)
        assert not tf.reduce_all(tf.abs(routing_output - softmax_output) < 1e-6)


class TestMultiHeadCrossAttention:
    """Test suite for MultiHeadCrossAttention layer."""

    @pytest.fixture
    def query_input(self):
        """Create a test query input tensor."""
        return tf.random.normal([2, 10, 64])  # (batch, query_seq_len, dim)

    @pytest.fixture
    def kv_input(self):
        """Create a test key-value input tensor."""
        return tf.random.normal([2, 20, 64])  # (batch, kv_seq_len, dim)

    # ==================== Initialization Tests ====================

    def test_initialization_defaults(self):
        """Test initialization with default parameters."""
        layer = MultiHeadCrossAttention(dim=64, num_heads=8)

        assert layer.dim == 64
        assert layer.num_heads == 8
        assert layer.head_dim == 8
        assert layer.dropout_rate == 0.0
        assert layer.shared_qk_projections is False
        assert layer.use_bias is True
        assert layer.use_adaptive_softmax is False
        assert layer.use_hierarchical_routing is False
        assert isinstance(layer.kernel_initializer, keras.initializers.GlorotUniform)
        assert layer.adaptive_softmax is None
        assert layer.hierarchical_routing is None

    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        custom_regularizer = keras.regularizers.L2(1e-4)
        adaptive_config = {"min_temp": 0.05, "max_temp": 5.0}

        layer = MultiHeadCrossAttention(
            dim=128,
            num_heads=16,
            dropout_rate=0.1,
            kernel_initializer="he_normal",
            kernel_regularizer=custom_regularizer,
            use_bias=False,
            use_adaptive_softmax=True,
            adaptive_softmax_config=adaptive_config
        )

        assert layer.dim == 128
        assert layer.num_heads == 16
        assert layer.dropout_rate == 0.1
        assert layer.use_bias is False
        assert layer.use_adaptive_softmax is True
        assert isinstance(layer.kernel_initializer, keras.initializers.HeNormal)
        assert layer.kernel_regularizer == custom_regularizer
        assert layer.adaptive_softmax is not None

    def test_initialization_hierarchical_routing(self):
        """Test initialization with hierarchical routing enabled."""
        layer = MultiHeadCrossAttention(
            dim=64,
            num_heads=8,
            use_hierarchical_routing=True
        )

        assert layer.use_hierarchical_routing is True
        assert layer.hierarchical_routing is not None
        assert isinstance(layer.hierarchical_routing, RoutingProbabilitiesLayer)

    def test_invalid_dim_not_divisible(self):
        """Test that invalid dim raises ValueError."""
        with pytest.raises(ValueError, match="dim \\(63\\) must be divisible by num_heads \\(8\\)"):
            MultiHeadCrossAttention(dim=63, num_heads=8)

    def test_invalid_adaptive_softmax_config(self):
        """Test invalid parameters for adaptive softmax."""
        with pytest.raises(ValueError, match="min_temp must be positive"):
            MultiHeadCrossAttention(
                dim=64,
                num_heads=8,
                use_adaptive_softmax=True,
                adaptive_softmax_config={"min_temp": 0}
            )

    def test_mutually_exclusive_normalization(self):
        """Test that both adaptive softmax and hierarchical routing can be used."""
        # Both should be allowed independently
        layer1 = MultiHeadCrossAttention(
            dim=64,
            num_heads=8,
            use_adaptive_softmax=True
        )
        assert layer1.use_adaptive_softmax is True
        assert layer1.use_hierarchical_routing is False

        layer2 = MultiHeadCrossAttention(
            dim=64,
            num_heads=8,
            use_hierarchical_routing=True
        )
        assert layer2.use_adaptive_softmax is False
        assert layer2.use_hierarchical_routing is True

    # ==================== Build Process Tests ====================

    def test_build_cross_attention(self, query_input, kv_input):
        """Test build process for cross-attention mode."""
        layer = MultiHeadCrossAttention(dim=64, num_heads=8)
        layer(query_input, kv_input)
        assert layer.built is True
        assert layer.q_dense is not None and layer.q_dense.built
        assert layer.kv_dense is not None and layer.kv_dense.built
        assert layer.qkv_dense is None

    def test_build_self_attention_shared(self, query_input):
        """Test build process for self-attention with shared projections."""
        layer = MultiHeadCrossAttention(dim=64, num_heads=8, shared_qk_projections=True)
        layer(query_input)
        assert layer.built is True
        assert layer.qkv_dense is not None and layer.qkv_dense.built
        assert layer.q_dense is None and layer.kv_dense is None

    def test_build_hierarchical_routing(self, query_input, kv_input):
        """Test build process with hierarchical routing."""
        layer = MultiHeadCrossAttention(
            dim=64,
            num_heads=8,
            use_hierarchical_routing=True
        )
        output = layer(query_input, kv_input)

        # After calling the layer, hierarchical routing should be built
        assert layer.hierarchical_routing.built is True
        # Check output shape is correct
        assert output.shape == query_input.shape

    # ==================== Output Shape Tests ====================

    def test_output_shape_cross_attention(self, query_input, kv_input):
        """Test output shape for cross-attention."""
        layer = MultiHeadCrossAttention(dim=64, num_heads=8)
        output = layer(query_input, kv_input)
        assert output.shape == query_input.shape

    def test_output_shape_self_attention(self, query_input):
        """Test output shape for self-attention."""
        layer = MultiHeadCrossAttention(dim=64, num_heads=8)
        output = layer(query_input)
        assert output.shape == query_input.shape

    # ==================== Forward Pass Tests ====================

    def test_forward_pass_hierarchical_routing(self, query_input, kv_input):
        """Test forward pass with hierarchical routing."""
        layer = MultiHeadCrossAttention(
            dim=64,
            num_heads=8,
            use_hierarchical_routing=True,
            dropout_rate=0.0
        )
        output = layer(query_input, kv_input, training=False)

        assert output.shape == query_input.shape
        assert not tf.reduce_any(tf.math.is_nan(output))
        assert not tf.reduce_any(tf.math.is_inf(output))

    def test_forward_pass_adaptive_softmax(self, query_input, kv_input):
        """Test forward pass with adaptive softmax."""
        layer = MultiHeadCrossAttention(
            dim=64,
            num_heads=8,
            use_adaptive_softmax=True
        )
        output = layer(query_input, kv_input)

        assert output.shape == query_input.shape
        assert not tf.reduce_any(tf.math.is_nan(output))

    def test_forward_pass_standard_softmax(self, query_input, kv_input):
        """Test forward pass with standard softmax."""
        layer = MultiHeadCrossAttention(dim=64, num_heads=8)
        output = layer(query_input, kv_input)

        assert output.shape == query_input.shape
        assert not tf.reduce_any(tf.math.is_nan(output))

    def test_different_attention_mechanisms_produce_different_outputs(
        self, query_input, kv_input
    ):
        """Test that different attention mechanisms produce different outputs."""
        tf.random.set_seed(42)
        layer_standard = MultiHeadCrossAttention(dim=64, num_heads=8, dropout_rate=0.0)
        output_standard = layer_standard(query_input, kv_input, training=False)

        tf.random.set_seed(42)
        layer_routing = MultiHeadCrossAttention(
            dim=64,
            num_heads=8,
            use_hierarchical_routing=True,
            dropout_rate=0.0
        )
        output_routing = layer_routing(query_input, kv_input, training=False)

        # Outputs should be different
        assert not tf.reduce_all(
            tf.abs(output_standard - output_routing) < 1e-6
        ), "Different attention mechanisms should produce different outputs"

    def test_shared_projections_with_kv_input_fails(self, query_input, kv_input):
        """Test that shared_qk_projections with kv_input raises an error."""
        layer = MultiHeadCrossAttention(dim=64, num_heads=8, shared_qk_projections=True)
        with pytest.raises(ValueError, match="When `shared_qk_projections=True`"):
            layer(query_input, kv_input)

    # ==================== Attention Mask Tests ====================

    def test_padding_mask_with_routing(self, query_input, kv_input):
        """Test padding mask with hierarchical routing."""
        layer = MultiHeadCrossAttention(
            dim=64,
            num_heads=8,
            use_hierarchical_routing=True,
            dropout_rate=0.0
        )

        mask = tf.ones((kv_input.shape[0], kv_input.shape[1]))
        mask = tf.concat([mask[:, :-5], tf.zeros((kv_input.shape[0], 5))], axis=1)

        output_masked = layer(query_input, kv_input, attention_mask=mask, training=False)
        output_unmasked = layer(query_input, kv_input, training=False)

        assert not tf.reduce_all(tf.equal(output_masked, output_unmasked))
        assert not tf.reduce_any(tf.math.is_nan(output_masked))

    def test_full_attention_mask(self, query_input, kv_input):
        """Test with a 3D attention mask."""
        layer = MultiHeadCrossAttention(dim=64, num_heads=8, dropout_rate=0.0)
        mask = tf.ones((query_input.shape[0], query_input.shape[1], kv_input.shape[1]))
        mask_np = mask.numpy()
        mask_np[:, 0, 0] = 0
        mask = tf.constant(mask_np)

        output_masked = layer(query_input, kv_input, attention_mask=mask, training=False)
        output_unmasked = layer(query_input, kv_input, training=False)
        assert not tf.reduce_all(tf.equal(output_masked, output_unmasked))

    # ==================== Serialization Tests ====================

    def test_serialization_hierarchical_routing(self):
        """Test serialization with hierarchical routing."""
        layer = MultiHeadCrossAttention(
            dim=128,
            num_heads=8,
            use_hierarchical_routing=True
        )

        config = layer.get_config()
        assert "use_hierarchical_routing" in config
        assert config["use_hierarchical_routing"] is True

        recreated_layer = MultiHeadCrossAttention.from_config(config)
        assert recreated_layer.use_hierarchical_routing is True
        assert recreated_layer.hierarchical_routing is not None

    def test_serialization_all_features(self):
        """Test serialization with all features enabled."""
        layer = MultiHeadCrossAttention(
            dim=256,
            num_heads=16,
            dropout_rate=0.2,
            use_adaptive_softmax=True,
            adaptive_softmax_config={"min_temp": 0.1, "max_temp": 2.0}
        )

        config = layer.get_config()
        recreated_layer = MultiHeadCrossAttention.from_config(config)

        assert recreated_layer.dim == 256
        assert recreated_layer.num_heads == 16
        assert recreated_layer.use_adaptive_softmax is True

    # ==================== Model Integration Tests ====================

    def test_model_with_hierarchical_routing(self, query_input, kv_input):
        """Test model integration with hierarchical routing."""
        query = keras.Input(shape=query_input.shape[1:])
        kv = keras.Input(shape=kv_input.shape[1:])
        x = MultiHeadCrossAttention(
            dim=64,
            num_heads=8,
            use_hierarchical_routing=True,
            name="routing_attention"
        )(query, kv)
        x = keras.layers.GlobalAveragePooling1D()(x)
        outputs = keras.layers.Dense(10)(x)

        model = keras.Model(inputs=[query, kv], outputs=outputs)
        model.compile(optimizer="adam", loss="mse")

        y_pred = model([query_input, kv_input], training=False)
        assert y_pred.shape == (query_input.shape[0], 10)
        assert not tf.reduce_any(tf.math.is_nan(y_pred))

    def test_model_save_load_with_routing(self, query_input, kv_input):
        """Test saving and loading model with hierarchical routing."""
        query = keras.Input(shape=query_input.shape[1:])
        kv = keras.Input(shape=kv_input.shape[1:])
        x = MultiHeadCrossAttention(
            dim=64,
            num_heads=8,
            use_hierarchical_routing=True,
            name="routing_attention"
        )(query, kv)
        outputs = keras.layers.GlobalAveragePooling1D()(x)

        model = keras.Model(inputs=[query, kv], outputs=outputs)
        original_prediction = model.predict([query_input, kv_input], verbose=0)

        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, "model.keras")
            model.save(model_path)

            loaded_model = keras.models.load_model(
                model_path,
                custom_objects={
                    "MultiHeadCrossAttention": MultiHeadCrossAttention,
                    "RoutingProbabilitiesLayer": RoutingProbabilitiesLayer
                }
            )

            loaded_prediction = loaded_model.predict([query_input, kv_input], verbose=0)

            np.testing.assert_allclose(
                original_prediction, loaded_prediction,
                rtol=1e-5, atol=1e-5,
                err_msg="Predictions should match after save/load"
            )

    # ==================== Gradient Flow Tests ====================

    def test_gradient_flow_hierarchical_routing(self, query_input, kv_input):
        """Test gradient flow with hierarchical routing."""
        layer = MultiHeadCrossAttention(
            dim=64,
            num_heads=8,
            use_hierarchical_routing=True
        )

        with tf.GradientTape() as tape:
            q_var = tf.Variable(query_input)
            kv_var = tf.Variable(kv_input)
            outputs = layer(q_var, kv_var)
            loss = tf.reduce_mean(tf.square(outputs))

        grads = tape.gradient(loss, layer.trainable_variables)
        assert all(g is not None for g in grads), "All gradients should be non-None"

    # ==================== Edge Case Tests ====================

    def test_numerical_stability_routing(self):
        """Test numerical stability with routing and extreme values."""
        layer = MultiHeadCrossAttention(
            dim=64,
            num_heads=8,
            use_hierarchical_routing=True
        )

        test_cases = [
            tf.zeros((2, 10, 64)),
            tf.ones((2, 10, 64)) * 1e-10,
            tf.ones((2, 10, 64)) * 1e3,
        ]

        for i, test_input in enumerate(test_cases):
            output = layer(test_input, test_input)
            assert not tf.reduce_any(tf.math.is_nan(output)), f"NaN in test case {i}"
            assert not tf.reduce_any(tf.math.is_inf(output)), f"Inf in test case {i}"


class TestRoutingIntegration:
    """Test integration of routing layer in various contexts."""

    def test_routing_as_softmax_replacement(self):
        """Test routing layer as a drop-in softmax replacement."""
        inputs = keras.Input(shape=(20,))
        dense_logits = keras.layers.Dense(10)(inputs)

        # Model with softmax
        model_softmax = keras.Model(
            inputs=inputs,
            outputs=keras.layers.Softmax()(dense_logits)
        )

        # Model with routing
        model_routing = keras.Model(
            inputs=inputs,
            outputs=RoutingProbabilitiesLayer(output_dim=10)(dense_logits)
        )

        test_inputs = tf.random.normal([5, 20])

        output_softmax = model_softmax(test_inputs)
        output_routing = model_routing(test_inputs)

        # Both should produce valid probability distributions
        assert tf.reduce_all(output_softmax >= 0.0) and tf.reduce_all(output_softmax <= 1.0)
        assert tf.reduce_all(output_routing >= 0.0) and tf.reduce_all(output_routing <= 1.0)

        # Both should sum to 1
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(tf.reduce_sum(output_softmax, axis=-1)),
            np.ones(5),
            rtol=1e-5, atol=1e-5
        )
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(tf.reduce_sum(output_routing, axis=-1)),
            np.ones(5),
            rtol=1e-5, atol=1e-5
        )

    def test_attention_routing_comparison_shapes(self):
        """Test that routing produces same shapes as standard attention."""
        query = tf.random.normal([2, 10, 64])
        kv = tf.random.normal([2, 20, 64])

        attn_standard = MultiHeadCrossAttention(dim=64, num_heads=8, dropout_rate=0.0)
        attn_routing = MultiHeadCrossAttention(
            dim=64,
            num_heads=8,
            use_hierarchical_routing=True,
            dropout_rate=0.0
        )

        out_standard = attn_standard(query, kv, training=False)
        out_routing = attn_routing(query, kv, training=False)

        assert out_standard.shape == out_routing.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])