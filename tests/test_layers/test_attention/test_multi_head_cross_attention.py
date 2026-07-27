"""Test suite for MultiHeadCrossAttention and RoutingProbabilitiesLayer.

This module contains comprehensive tests for both the MultiHeadCrossAttention layer
and the RoutingProbabilitiesLayer, validating their functionality independently and
in integration, including cross-attention, self-attention, masking, the unified
``probability_type`` API (softmax / sparsemax / threshmax / adaptive), and
serialization.
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
        assert layer.kernel.shape == (15, 4)  # (input_dim, num_decisions)

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

        # Check shape: kernel is (input_dim, num_decisions)
        assert layer.kernel.shape[0] == 20
        assert layer.kernel.shape[1] == layer.num_decisions

        # Check that weights are normalized (unit L2 norm per decision column)
        for i in range(layer.num_decisions):
            weights = layer.kernel[:, i]
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
        # One non-trainable weight: the cosine basis. Per DECISION D-006 in
        # routing_probabilities.py, the leaf-validity masks (mask_mul,
        # mask_add) are deterministic functions of (output_dim,
        # padded_output_dim) and are stored as numpy arrays — NOT as
        # add_weight — to avoid checkpoint bloat. They are recomputed in
        # build() on every load and converted to backend tensors inside call().
        assert len(layer.non_trainable_variables) == 1, \
            "Deterministic mode should have one non-trainable weight (cosine basis)"

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
        assert layer.probability_type == "softmax"
        assert layer.probability_config is None
        assert layer.qk_norm_type is None
        assert layer.qk_norm_kwargs is None
        assert isinstance(layer.kernel_initializer, keras.initializers.GlorotUniform)
        assert layer.attn_prob is not None

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
            probability_type="adaptive",
            probability_config=adaptive_config,
        )

        assert layer.dim == 128
        assert layer.num_heads == 16
        assert layer.dropout_rate == 0.1
        assert layer.use_bias is False
        assert layer.probability_type == "adaptive"
        assert layer.probability_config == adaptive_config
        assert isinstance(layer.kernel_initializer, keras.initializers.HeNormal)
        assert layer.kernel_regularizer == custom_regularizer
        assert layer.attn_prob is not None

    def test_initialization_hierarchical_routing_rejected(self):
        """``probability_type='hierarchical'`` must be rejected.

        Routing/hierarchical strategies consume FEATURES and require a fixed
        ``output_dim``; they cannot operate on attention-score logits whose
        last dimension is the dynamic kv sequence length.
        """
        with pytest.raises(ValueError, match="not supported"):
            MultiHeadCrossAttention(
                dim=64,
                num_heads=8,
                probability_type="hierarchical",
            )

        with pytest.raises(ValueError, match="not supported"):
            MultiHeadCrossAttention(
                dim=64,
                num_heads=8,
                probability_type="routing",
            )

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
                probability_type="adaptive",
                probability_config={"min_temp": 0},
            )

    def test_alternative_probability_types(self):
        """Test alternative (non-softmax) probability types are accepted."""
        for ptype in ("softmax", "sparsemax", "threshmax", "adaptive"):
            layer = MultiHeadCrossAttention(
                dim=64,
                num_heads=8,
                probability_type=ptype,
            )
            assert layer.probability_type == ptype
            assert layer.attn_prob is not None

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

    def test_build_alternative_probability(self, query_input, kv_input):
        """Test build process with an alternative (sparsemax) probability type."""
        layer = MultiHeadCrossAttention(
            dim=64,
            num_heads=8,
            probability_type="sparsemax",
        )
        output = layer(query_input, kv_input)

        # After calling the layer, attn_prob should be built
        assert layer.attn_prob.built is True
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

    def test_forward_pass_sparsemax(self, query_input, kv_input):
        """Test forward pass with sparsemax probability type."""
        layer = MultiHeadCrossAttention(
            dim=64,
            num_heads=8,
            probability_type="sparsemax",
            dropout_rate=0.0,
        )
        output = layer(query_input, kv_input, training=False)

        assert output.shape == query_input.shape
        assert not tf.reduce_any(tf.math.is_nan(output))
        assert not tf.reduce_any(tf.math.is_inf(output))

    def test_forward_pass_threshmax(self, query_input, kv_input):
        """Test forward pass with threshmax probability type."""
        layer = MultiHeadCrossAttention(
            dim=64,
            num_heads=8,
            probability_type="threshmax",
            dropout_rate=0.0,
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
            probability_type="adaptive",
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
        layer_alt = MultiHeadCrossAttention(
            dim=64,
            num_heads=8,
            probability_type="sparsemax",
            dropout_rate=0.0,
        )
        output_alt = layer_alt(query_input, kv_input, training=False)

        # Outputs should be different
        assert not tf.reduce_all(
            tf.abs(output_standard - output_alt) < 1e-6
        ), "Different attention mechanisms should produce different outputs"

    def test_shared_projections_with_kv_input_fails(self, query_input, kv_input):
        """Test that shared_qk_projections with kv_input raises an error."""
        layer = MultiHeadCrossAttention(dim=64, num_heads=8, shared_qk_projections=True)
        with pytest.raises(ValueError, match="When `shared_qk_projections=True`"):
            layer(query_input, kv_input)

    # ==================== Attention Mask Tests ====================

    def test_padding_mask_with_sparsemax(self, query_input, kv_input):
        """Test padding mask with sparsemax probability type."""
        layer = MultiHeadCrossAttention(
            dim=64,
            num_heads=8,
            probability_type="sparsemax",
            dropout_rate=0.0,
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

    def test_serialization_sparsemax(self):
        """Test serialization round-trip with sparsemax probability type."""
        layer = MultiHeadCrossAttention(
            dim=128,
            num_heads=8,
            probability_type="sparsemax",
        )

        config = layer.get_config()
        assert "probability_type" in config
        assert config["probability_type"] == "sparsemax"

        recreated_layer = MultiHeadCrossAttention.from_config(config)
        assert recreated_layer.probability_type == "sparsemax"
        assert recreated_layer.attn_prob is not None

    def test_serialization_all_features(self):
        """Test serialization with adaptive probability type and config."""
        layer = MultiHeadCrossAttention(
            dim=256,
            num_heads=16,
            dropout_rate=0.2,
            probability_type="adaptive",
            probability_config={"min_temp": 0.1, "max_temp": 2.0},
        )

        config = layer.get_config()
        recreated_layer = MultiHeadCrossAttention.from_config(config)

        assert recreated_layer.dim == 256
        assert recreated_layer.num_heads == 16
        assert recreated_layer.probability_type == "adaptive"
        assert recreated_layer.probability_config == {"min_temp": 0.1, "max_temp": 2.0}

    # ==================== Model Integration Tests ====================

    def test_model_with_sparsemax(self, query_input, kv_input):
        """Test model integration with sparsemax probability type."""
        query = keras.Input(shape=query_input.shape[1:])
        kv = keras.Input(shape=kv_input.shape[1:])
        x = MultiHeadCrossAttention(
            dim=64,
            num_heads=8,
            probability_type="sparsemax",
            name="sparsemax_attention",
        )(query, kv)
        x = keras.layers.GlobalAveragePooling1D()(x)
        outputs = keras.layers.Dense(10)(x)

        model = keras.Model(inputs=[query, kv], outputs=outputs)
        model.compile(optimizer="adam", loss="mse")

        y_pred = model([query_input, kv_input], training=False)
        assert y_pred.shape == (query_input.shape[0], 10)
        assert not tf.reduce_any(tf.math.is_nan(y_pred))

    def test_model_save_load_with_sparsemax(self, query_input, kv_input):
        """Test saving and loading a model that uses sparsemax attention."""
        query = keras.Input(shape=query_input.shape[1:])
        kv = keras.Input(shape=kv_input.shape[1:])
        x = MultiHeadCrossAttention(
            dim=64,
            num_heads=8,
            probability_type="sparsemax",
            name="sparsemax_attention",
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

    def test_gradient_flow_sparsemax(self, query_input, kv_input):
        """Test gradient flow with sparsemax probability type."""
        layer = MultiHeadCrossAttention(
            dim=64,
            num_heads=8,
            probability_type="sparsemax",
        )

        with tf.GradientTape() as tape:
            q_var = tf.Variable(query_input)
            kv_var = tf.Variable(kv_input)
            outputs = layer(q_var, kv_var)
            loss = tf.reduce_mean(tf.square(outputs))

        grads = tape.gradient(loss, layer.trainable_variables)
        assert all(g is not None for g in grads), "All gradients should be non-None"

    # ==================== Edge Case Tests ====================

    def test_numerical_stability_sparsemax(self):
        """Test numerical stability with sparsemax and extreme values."""
        layer = MultiHeadCrossAttention(
            dim=64,
            num_heads=8,
            probability_type="sparsemax",
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
        attn_alt = MultiHeadCrossAttention(
            dim=64,
            num_heads=8,
            probability_type="sparsemax",
            dropout_rate=0.0,
        )

        out_standard = attn_standard(query, kv, training=False)
        out_alt = attn_alt(query, kv, training=False)

        assert out_standard.shape == out_alt.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# ---------------------------------------------------------------------
# Mixed-precision mask tests (plan-2026-07-27-b4ef45f0, step 4)
# ---------------------------------------------------------------------
#
# WHAT IS BEING GUARDED HERE.
#
# `MultiHeadCrossAttention._apply_attention_mask` used the ARITHMETIC mask form
#
#     return scores + (1.0 - attention_mask) * mask_value   # mask_value = -1e9
#
# which `common.MASK_BIAS_VALUE`'s own docstring rules out. Under
# `mixed_precision.set_global_policy('mixed_float16')` the literal `-1e9` is
# materialized in float16, where it is `-inf` (np.float16(-1e9) == -inf). At every
# UNMASKED position `(1.0 - mask) == 0`, so the product is `0 * -inf = NaN` — the
# NaN appears exactly where NOTHING was masked, and the following matmul spreads it
# across the whole batch.
#
# MEASURED on unfixed HEAD (B=2, N=64, D=64, num_heads=4), GPU 1 / TF 2.18,
# non-finite entries in the layer OUTPUT:
#
#     policy            no mask   all-ones mask   padding mask   causal mask
#     float32            0/8192      0/8192          0/8192        0/8192
#     mixed_float16      0/8192   8192/8192       8192/8192     8192/8192
#
# The all-ones column is the important one: a mask that masks NOTHING destroys the
# entire batch. That is not a pathological input — it is what a caller passes when
# every sequence in the batch happens to be full length.
#
# THE FIX is `common.apply_attention_mask`, which builds the bias with `ops.where`
# inside `common.mask_dtype(...)` (>= float32), so `0 * -inf` cannot be formed at
# all. Each site keeps its own broadcast/cast order and its own polarity spelling.
#
# WHAT THIS FIX DOES **NOT** COVER (assumption A2, measured at step 4): a FULLY-MASKED
# query row. The biased logits are cast back to the compute dtype at this site, so in
# fp16 an all-masked row is all-`-inf` again and `softmax(all -inf)` is `0/0 = NaN`.
# Casting back is not the problem and `out_dtype=None` would not help: the softmax
# here is `self.attn_prob`, a Keras layer with autocasting ON, which drags a float32
# tensor straight back to float16 (pinned by
# `TestMultiHeadCrossAttentionMaskHazardIsReal::test_the_probability_sublayer_autocasts_a_float32_input`).
# Removing that failure mode needs the predicate-level rescue used in
# `capsule_routing_attention.py` (D-006), which is a SEMANTICS change on a degenerate
# input and is deliberately not part of this step. What IS guaranteed, and asserted
# below, is that the damage no longer spreads: every row that keeps >= 1 key stays
# finite. Before the fix, one degenerate row NaN'd all 8192 outputs.
#
# ANTI-VACUITY. The `N = 7`-hides-an-fp16-`-inf`-at-`N >= 512` trap does not transfer:
# this hazard is a per-ELEMENT dtype overflow of a constant multiplied by an exact
# zero, not a long reduction, so it appears at any N >= 1. It is nevertheless
# asserted reachable rather than assumed — see `TestMultiHeadCrossAttentionMaskHazardIsReal`, which
# checks that the policy really selects float16 compute, that
# `float16(MASK_BIAS_VALUE)` really is `-inf`, that `float16(0) * that` really is
# `NaN`, and that each mask really has the structure its name claims. N = 64 keys is
# a realistic sequence length for this layer rather than a toy one.

from dl_techniques.layers.attention.common import MASK_BIAS_VALUE

_MP_B, _MP_N, _MP_D, _MP_H = 2, 64, 64, 4
_MP_KEEP = _MP_N // 2            # first half kept, second half masked (padding mask)
_MP_DEG_ROW = 5                  # the query row the 'degenerate' mask blanks entirely
_MP_SEED = 1234

# Absolute tolerance for "this policy's masked forward agrees with the float32 one".
#
# PRE-REGISTERED, not tuned after the fact: sized from the layer's own NO-MASK dtype
# error measured on unfixed HEAD (the only forward that survives fp16 there), which is
# the honest budget — a correct mask fix should leave the masked path no worse
# conditioned than the unmasked one. Measured no-mask max |policy - float32|:
# mixed_float16 0.0051 and float64 0.0078, against an output absmax of 6.73 (up to
# 7.73 once a mask is applied). The entries below carry ~10x headroom on that.
# float32 compares against a control computed the same way and is exact.
_MP_ATOL = {"float32": 1e-6, "mixed_float16": 0.05, "float64": 0.05}


def _mp_input():
    """Deterministic ``(B, N, D)`` float32 input, shared by every test below."""
    return np.random.default_rng(7).standard_normal(
        (_MP_B, _MP_N, _MP_D)
    ).astype("float32")


def _mp_mask(kind):
    """One of the masks these tests need, as a float32 ``1 = keep`` array.

    ``'all_ones'`` masks NOTHING and is the catastrophic case for the arithmetic
    form. ``'padding'`` masks the second half of the keys with a rank-2 ``(B, N)`` mask, exercising this site's rank-2 broadcast branch, ``'causal'`` is
    lower-triangular, and ``'degenerate'`` blanks query row ``_MP_DEG_ROW`` entirely
    (the A2 probe — see the note above; it is NOT part of the finiteness contract
    this step establishes).
    """
    if kind == "all_ones":
        return np.ones((_MP_B, _MP_N, _MP_N), dtype="float32")
    if kind == "padding":
        m = np.ones((_MP_B, _MP_N), dtype="float32")
        m[:, _MP_KEEP:] = 0.0
        return m
    if kind == "causal":
        return np.broadcast_to(
            np.tril(np.ones((_MP_N, _MP_N), dtype="float32")), (_MP_B, _MP_N, _MP_N)
        ).copy()
    if kind == "degenerate":
        m = np.ones((_MP_B, _MP_N, _MP_N), dtype="float32")
        m[:, _MP_DEG_ROW, :] = 0.0
        return m
    raise ValueError(f"unknown mask kind {kind!r}")


def _mp_layer(**kwargs):
    """A built layer whose TRAINABLE weights are identical under every dtype policy.

    Seeding the initializers is NOT sufficient: a ``glorot_uniform`` draw under a
    ``float64`` policy differs from the same-seed draw under ``float32`` (the
    initializer samples in the VARIABLE dtype), so a cross-policy comparison on
    seeded-but-not-assigned weights measures the initializer, not the code under
    test. Explicit values are assigned instead. Non-trainable buffers (e.g. the RoPE
    cos/sin cache) are left as the layer computes them.
    """
    layer = MultiHeadCrossAttention(dim=_MP_D, num_heads=_MP_H, **kwargs)
    layer.build((_MP_B, _MP_N, _MP_D))
    rng = np.random.default_rng(_MP_SEED)
    for weight in layer.trainable_weights:
        shape = tuple(weight.shape)
        if len(shape) == 1 and ("bias" in weight.name or "beta" in weight.name):
            value = np.zeros(shape)
        elif len(shape) == 1:                     # a scale / gamma: keep it near 1
            value = 1.0 + 0.1 * rng.standard_normal(shape)
        else:
            value = 0.2 * rng.standard_normal(shape)
        weight.assign(keras.ops.cast(
            keras.ops.convert_to_tensor(value.astype("float32")), weight.dtype
        ))
    return layer


def _mp_forward(layer, array, mask):
    """One masked forward pass, returned as float64 numpy."""
    out = layer(keras.ops.convert_to_tensor(array), attention_mask=(None if mask is None else keras.ops.convert_to_tensor(mask)))
    if isinstance(out, (list, tuple)):
        out = out[0]
    return keras.ops.convert_to_numpy(out).astype("float64")


_F32_REFERENCE = {}


def _float32_reference(kind):
    """Masked float32 output for ``kind``, memoized, under an explicit policy.

    This is the CONTROL every mixed-precision assertion compares against. It sets and
    restores the policy itself, so it is valid whichever parametrization of
    ``dtype_policy`` happens to reach it first.
    """
    if kind not in _F32_REFERENCE:
        previous = keras.mixed_precision.global_policy().name
        keras.mixed_precision.set_global_policy("float32")
        try:
            layer = _mp_layer()
            _F32_REFERENCE[kind] = _mp_forward(
                layer, _mp_input(), _mp_mask(kind)
            )
        finally:
            keras.mixed_precision.set_global_policy(previous)
    return _F32_REFERENCE[kind]


class TestMultiHeadCrossAttentionMaskHazardIsReal:
    """Anti-vacuity. If these stop holding, every fp16 test below is worthless."""

    def test_policy_really_selects_float16_compute(self, dtype_policy):
        expected = {
            "float32": "float32",
            "mixed_float16": "float16",
            "float64": "float64",
        }[dtype_policy]
        assert keras.mixed_precision.global_policy().compute_dtype == expected

    def test_the_arithmetic_form_really_is_nan_in_the_compute_dtype(self):
        with np.errstate(over="ignore", invalid="ignore"):
            bias = np.float16(MASK_BIAS_VALUE)
            assert np.isneginf(bias), (
                "anti-vacuity FAILED: float16(MASK_BIAS_VALUE) is not -inf, so the "
                "`0 * -inf` hazard this module guards is not reproducible here."
            )
            assert np.isnan(np.float16(0.0) * bias), (
                "anti-vacuity FAILED: float16(0) * float16(MASK_BIAS_VALUE) is not "
                "NaN — the arithmetic mask form would be harmless."
            )
        assert np.isfinite(np.float32(MASK_BIAS_VALUE)), (
            "anti-vacuity FAILED: MASK_BIAS_VALUE is not finite in float32, so "
            "`mask_dtype(...)` would not be a fix."
        )

    def test_the_all_ones_mask_really_masks_nothing(self):
        mask = _mp_mask("all_ones")
        assert int((mask == 0).sum()) == 0, (
            "the 'all_ones' mask masks something; it no longer reproduces the "
            "signature catastrophe (a vacuous mask destroying the batch)"
        )

    def test_the_partial_masks_really_mask_something(self):
        for kind in ("padding", "causal"):
            mask = _mp_mask(kind)
            assert int((mask == 0).sum()) > 0, (
                f"the {kind!r} mask masks nothing; it cannot detect a regression "
                "in the masking code"
            )
            rows = mask if mask.ndim == 3 else mask[:, None, :]
            assert not (rows == 0).all(axis=-1).any(), (
                f"the {kind!r} mask has a fully-masked query row, so it no longer "
                "isolates the covered case from the A2 probe"
            )

    def test_the_degenerate_mask_really_has_exactly_one_fully_masked_row(self):
        mask = _mp_mask("degenerate")
        empty = (mask == 0).all(axis=-1)
        assert int(empty.sum()) == _MP_B, (
            f"expected exactly one fully-masked query row per batch element, got "
            f"{int(empty.sum())} across {_MP_B} batch elements"
        )
        assert empty[:, _MP_DEG_ROW].all()

    def test_the_probability_sublayer_autocasts_a_float32_input(self, dtype_policy):
        """Why ``out_dtype`` cannot rescue a fully-masked row at this site.

        The softmax here is a Keras LAYER with autocasting on, so handing it a
        carefully-promoted float32 tensor changes nothing — it is seen inside its
        own ``call()`` as the compute dtype. This is the same measurement that
        selected the predicate-level rescue in `capsule_routing_attention.py`
        (assumption A4). If Keras ever stops doing this, this test fails and the
        ``out_dtype`` choice at this site can be revisited.
        """
        layer = _mp_layer()
        prob = layer.attn_prob
        assert getattr(prob, "autocast", False) is True

        seen = {}
        original = prob.call

        def spy(x, *args, **kwargs):
            seen["dtype"] = keras.backend.standardize_dtype(x.dtype)
            return original(x, *args, **kwargs)

        prob.call = spy
        try:
            prob(keras.ops.convert_to_tensor(np.zeros((1, _MP_H, 4, 4), dtype="float32")))
        finally:
            prob.call = original

        expected = keras.mixed_precision.global_policy().compute_dtype
        assert seen["dtype"] == expected, (
            f"a float32 tensor entering `attn_prob` was seen inside its call() as "
            f"{seen['dtype']!r}, not the compute dtype {expected!r}"
        )


class TestMultiHeadCrossAttentionMixedPrecisionMask:
    """SC1 + SC2: finite AND agreeing with float32, for every legal mask."""

    @pytest.mark.parametrize("kind", ["all_ones", "padding", "causal"])
    def test_masked_forward_is_finite_and_matches_float32(self, dtype_policy, kind):

        layer = _mp_layer()
        out = _mp_forward(layer, _mp_input(), _mp_mask(kind))

        n_bad = int((~np.isfinite(out)).sum())
        assert n_bad == 0, (
            f"{n_bad}/{out.size} non-finite output entries for a {kind!r} mask "
            f"under policy {dtype_policy!r}"
        )

        reference = _float32_reference(kind)
        atol = _MP_ATOL[dtype_policy]
        max_dev = float(np.abs(out - reference).max())
        assert max_dev <= atol, (
            f"{kind!r}-masked forward under {dtype_policy!r} deviates from the "
            f"float32 control by {max_dev:.4g} > {atol:.4g}"
        )
        assert float(np.abs(out).max()) > 0.5 * float(np.abs(reference).max()), (
            f"output absmax {np.abs(out).max():.4g} collapsed relative to the "
            f"float32 control {np.abs(reference).max():.4g}"
        )


class TestMultiHeadCrossAttentionFullyMaskedRow:
    """Assumption A2 at this site, as an executable statement of what is guaranteed.

    A2 predicted that casting the biased logits back to the compute dtype is safe
    here because every softmax row keeps at least one position. A caller CAN break
    that by masking a whole query row, so the boundary is measured rather than
    assumed. What this step guarantees — and what is asserted — is CONTAINMENT: rows
    that keep >= 1 key stay finite even when a sibling row is degenerate. Before the
    fix, one degenerate row made all 8192 outputs NaN.

    The degenerate row's own value is deliberately NOT compared against the float32
    control: in float32/float64 it is a uniform, meaningless distribution over all
    keys (garbage in, garbage out), and its numeric value is genuinely dtype-
    dependent. Fixing it needs the predicate-level rescue of
    `capsule_routing_attention.py` (decisions.md D-006), which is a semantics change
    outside this step.
    """

    def test_a_fully_masked_row_does_not_poison_the_rest_of_the_batch(
        self, dtype_policy
    ):

        layer = _mp_layer()
        out = _mp_forward(layer, _mp_input(), _mp_mask("degenerate"))

        kept = np.delete(out, _MP_DEG_ROW, axis=1)
        n_bad = int((~np.isfinite(kept)).sum())
        assert n_bad == 0, (
            f"{n_bad}/{kept.size} non-finite entries in the rows that KEEP keys, "
            f"under policy {dtype_policy!r} — a single degenerate query row is "
            "still poisoning the whole batch"
        )

        reference = np.delete(_float32_reference("degenerate"), _MP_DEG_ROW, axis=1)
        atol = _MP_ATOL[dtype_policy]
        max_dev = float(np.abs(kept - reference).max())
        assert max_dev <= atol, (
            f"the non-degenerate rows under {dtype_policy!r} deviate from the "
            f"float32 control by {max_dev:.4g} > {atol:.4g}"
        )

        if dtype_policy != "mixed_float16":
            assert np.isfinite(out[:, _MP_DEG_ROW]).all(), (
                f"the fully-masked query row is not finite under {dtype_policy!r}, "
                "where MASK_BIAS_VALUE is representable — that is a regression, not "
                "the known fp16 boundary"
            )


class TestMultiHeadCrossAttentionMaskPolarity:
    """SC6: the mask must suppress the MASKED positions, not the kept ones.

    A polarity inversion at this site — passing the keep predicate where its
    complement is meant, or vice versa — raises nothing, changes no shape and leaves
    the output perfectly finite. Only an influence test can see it. MEASURED on
    unmodified HEAD by handing the layer ``1 - mask``: perturbing a "masked" token
    then moves the kept query rows by 26.0 instead of 0.0, against a
    kept-token influence of 29.5.

    The statement is EXACT here (not approximate): a masked key contributes exactly
    `exp(-1e9) == 0` weight, so a perturbation of a masked token cannot reach a kept
    query row at all. Measured 0.0 in float32 on unfixed HEAD, so a real inversion
    is separated from correct behavior by the full 26.0 signal.
    """

    @staticmethod
    def _influence(layer, mask):
        base_input = _mp_input()
        perturbed_masked = base_input.copy()
        perturbed_masked[:, _MP_KEEP + 3, :] += 5.0      # a MASKED token
        perturbed_kept = base_input.copy()
        perturbed_kept[:, 3, :] += 5.0                   # a KEPT token

        rows = slice(0, _MP_KEEP)
        base = _mp_forward(layer, base_input, mask)
        assert np.isfinite(base[:, rows]).all(), (
            "the kept query rows are not finite; the comparison below would be "
            "meaningless"
        )
        delta_masked = float(
            np.abs(_mp_forward(layer, perturbed_masked, mask)[:, rows]
                   - base[:, rows]).max()
        )
        delta_kept = float(
            np.abs(_mp_forward(layer, perturbed_kept, mask)[:, rows]
                   - base[:, rows]).max()
        )
        return delta_masked, delta_kept

    def test_a_masked_token_has_no_influence_on_the_kept_rows(self, dtype_policy):

        layer = _mp_layer()
        delta_masked, delta_kept = self._influence(layer, _mp_mask("padding"))

        # Measured EXACTLY 0.0 on unfixed float32. The 1e-3 budget is session-noise
        # headroom (see `test_rpc_attention.py`, where a batched op measured 0.0 in
        # isolation and 1.1e-06 inside the full suite).
        assert delta_masked <= 1e-3, (
            f"perturbing a MASKED token changed the kept query rows by "
            f"{delta_masked:.6g} under policy {dtype_policy!r} — this must be "
            f"exact, so the mask polarity is INVERTED (the layer is attending to "
            f"the padding; measured 26.0 with a deliberately inverted mask)"
        )
        assert delta_kept > 1.0, (
            f"perturbing a KEPT token changed the output by only {delta_kept:.6g}; "
            "the test is vacuous — the layer is ignoring its input"
        )
