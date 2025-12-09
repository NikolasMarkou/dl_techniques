import pytest
import numpy as np
import tensorflow as tf
import keras
import os
import tempfile

from dl_techniques.layers.activations.routing_probabilities_hierarchical import HierarchicalRoutingLayer


# ==============================================================================
# Test Suite
# ==============================================================================

class TestHierarchicalRoutingLayer:
    """
    Test suite for the refined HierarchicalRoutingLayer implementation.

    Covers initialization, shape inference (arbitrary ranks), probability logic,
    numerical stability, serialization, and training integration.
    """

    @pytest.mark.parametrize(
        "output_dim, expected_padded, expected_decisions",
        [
            (2, 2, 1),  # Minimum valid dimension
            (3, 4, 2),  # Small non-power-of-2
            (7, 8, 3),  # Just under power of 2
            (8, 8, 3),  # Exact power of 2
            (17, 32, 5),  # Just over power of 2
            (1000, 1024, 10),  # Large dimension
        ]
    )
    def test_initialization_attributes(self, output_dim, expected_padded, expected_decisions):
        """Test that internal tree attributes are calculated correctly."""
        layer = HierarchicalRoutingLayer(output_dim=output_dim)

        # Trigger build to calculate attributes that depend on build (if any moved there)
        # Note: In the refactor, these are calculated in build, but some are init-dependent.
        # We call build with a dummy shape to ensure all state is set.
        layer.build((None, 10))

        assert layer.output_dim == output_dim
        assert layer.padded_output_dim == expected_padded
        assert layer.num_decisions == expected_decisions

        # Verify it doesn't use the old sub-layer structure
        assert not hasattr(layer, 'decision_dense')
        assert hasattr(layer, 'kernel')
        assert hasattr(layer, 'bias')

    def test_invalid_initialization(self):
        """Test validation logic in __init__."""
        with pytest.raises(ValueError, match="must be an integer greater than 1"):
            HierarchicalRoutingLayer(output_dim=1)
        with pytest.raises(ValueError, match="axis' must be an integer"):
            HierarchicalRoutingLayer(output_dim=10, axis="invalid")

    @pytest.mark.parametrize("axis", [-1, 1])
    def test_build_weights(self, axis):
        """Test that trainable weights are created with correct shapes."""
        output_dim = 10
        input_dim = 16
        # Expected depth: log2(16) = 4 (padded 10 -> 16)
        expected_decisions = 4

        layer = HierarchicalRoutingLayer(output_dim=output_dim, axis=axis)

        # Shape depends on axis
        if axis == -1:
            input_shape = (None, input_dim)
        else:
            input_shape = (None, input_dim, 5)  # Axis 1 is input_dim

        layer.build(input_shape)

        assert layer.built
        assert len(layer.trainable_weights) == 2  # Kernel + Bias

        # Kernel shape: (input_dim, num_decisions)
        assert layer.kernel.shape == (input_dim, expected_decisions)
        # Bias shape: (num_decisions,)
        assert layer.bias.shape == (expected_decisions,)

    @pytest.mark.parametrize(
        "input_shape, output_dim, axis, expected_output_shape",
        [
            # Standard 2D: (Batch, Features) -> (Batch, Classes)
            ((4, 16), 10, -1, (4, 10)),

            # 3D Sequence: (Batch, Time, Features) -> (Batch, Time, Classes)
            ((2, 5, 16), 10, -1, (2, 5, 10)),

            # 3D Image-like with axis 1: (Batch, Channels, Height) -> (Batch, Classes, Height)
            ((2, 16, 8), 5, 1, (2, 5, 8)),

            # 4D Image: (Batch, H, W, Channels) -> (Batch, H, W, Classes)
            ((2, 8, 8, 32), 10, -1, (2, 8, 8, 10)),
        ]
    )
    def test_forward_pass_shapes(self, input_shape, output_dim, axis, expected_output_shape):
        """
        Test that the layer correctly handles arbitrary rank inputs and axes.
        This validates the transpose-flatten-reshape logic introduced in the refactor.
        """
        layer = HierarchicalRoutingLayer(output_dim=output_dim, axis=axis)
        inputs = tf.random.normal(input_shape)

        # Run forward pass
        outputs = layer(inputs)

        assert outputs.shape == expected_output_shape

        # Verify compute_output_shape matches actual output
        computed_shape = layer.compute_output_shape(input_shape)
        assert computed_shape == expected_output_shape

    def test_output_normalization(self):
        """Test that outputs sum to 1.0 along the routing axis."""
        batch_size = 4
        time_steps = 3
        input_dim = 8
        output_dim = 5  # Non-power of 2 to trigger renormalization

        layer = HierarchicalRoutingLayer(output_dim=output_dim)
        inputs = tf.random.normal((batch_size, time_steps, input_dim))

        outputs = layer(inputs)

        # Sum along last axis
        sums = tf.reduce_sum(outputs, axis=-1)
        assert np.allclose(sums, 1.0, atol=1e-6)

    def test_manual_logic_power_of_2(self):
        """
        Manually set weights to verify the binary tree logic for a power-of-2 case.
        No renormalization should occur.
        """
        input_dim = 1
        output_dim = 4  # Padded dim is 4, Depth = 2

        layer = HierarchicalRoutingLayer(output_dim=output_dim, use_bias=True, epsilon=0.0)
        inputs = tf.ones((1, input_dim))  # Input is 1.0
        layer.build((1, input_dim))

        # We want logits:
        # Decision 0 (Root): High positive -> Sigmoid ~ 1.0 -> Go Right
        # Decision 1 (L1): High negative -> Sigmoid ~ 0.0 -> Go Left
        #
        # Tree Path for leaves:
        # L0 (Left, Left): (1-p0)(1-p1) = 0 * 1 = 0
        # L1 (Left, Right): (1-p0)p1    = 0 * 0 = 0
        # L2 (Right, Left): p0(1-p1)    = 1 * 1 = 1  <-- Target
        # L3 (Right, Right): p0p1       = 1 * 0 = 0

        # Weights: [input_dim, num_decisions] -> [1, 2]
        # We set kernel to 0 and control via bias for simplicity
        kernel = np.zeros((input_dim, 2))
        bias = np.array([10.0, -10.0])  # Logits: +10, -10

        layer.set_weights([kernel, bias])

        preds = layer(inputs).numpy().flatten()

        # Sigmoid(10) ~= 0.99995, Sigmoid(-10) ~= 0.000045
        p0 = 1.0 / (1.0 + np.exp(-10.0))
        p1 = 1.0 / (1.0 + np.exp(10.0))  # sigmoid(-10)

        expected = np.array([
            (1 - p0) * (1 - p1),  # L0
            (1 - p0) * p1,  # L1
            p0 * (1 - p1),  # L2 (Dominant)
            p0 * p1  # L3
        ])

        assert np.allclose(preds, expected, atol=1e-5)
        # Verify L2 is indeed the winner
        assert np.argmax(preds) == 2

    def test_manual_logic_renormalization(self):
        """
        Verify logic when output_dim is not a power of 2.
        Padded dim=4, Output dim=3. The 4th leaf is discarded and others renormalized.
        """
        input_dim = 1
        output_dim = 3

        layer = HierarchicalRoutingLayer(output_dim=output_dim, epsilon=0.0)
        inputs = tf.ones((1, input_dim))
        layer.build((1, input_dim))

        # Force uniform probability across all 4 padded leaves
        # Logits = 0 -> Sigmoid = 0.5
        kernel = np.zeros((1, 2))
        bias = np.zeros((2,))
        layer.set_weights([kernel, bias])

        preds = layer(inputs).numpy().flatten()

        # Padded leaves would be [0.25, 0.25, 0.25, 0.25]
        # Slice first 3: [0.25, 0.25, 0.25] -> Sum = 0.75
        # Renormalize: 0.25 / 0.75 = 1/3

        expected = np.array([1 / 3, 1 / 3, 1 / 3])

        assert np.allclose(preds, expected, atol=1e-6)

    def test_numerical_stability(self):
        """Test that extreme logits don't cause NaNs due to epsilon clipping."""
        layer = HierarchicalRoutingLayer(output_dim=10, epsilon=1e-7)
        inputs = tf.ones((1, 5))
        layer.build(inputs.shape)

        # Set extreme weights to force 0.0 and 1.0 sigmoids
        kernel = np.ones(layer.kernel.shape) * 1000.0
        bias = np.zeros(layer.bias.shape)
        layer.set_weights([kernel, bias])

        outputs = layer(inputs)

        assert not np.any(np.isnan(outputs))
        assert np.all(outputs > 0.0)  # Should be at least epsilon-ish

    def test_serialization(self):
        """Test get_config and from_config."""
        layer = HierarchicalRoutingLayer(
            output_dim=5,
            axis=1,
            epsilon=1e-5,
            use_bias=False,
            name="serial_test"
        )

        config = layer.get_config()
        new_layer = HierarchicalRoutingLayer.from_config(config)

        assert new_layer.output_dim == 5
        assert new_layer.axis == 1
        assert new_layer.epsilon == 1e-5
        assert new_layer.use_bias is False
        assert new_layer.name == "serial_test"

    def test_training_integration(self):
        """Verify the layer works in a standard training loop (gradient flow)."""
        input_dim = 8
        output_dim = 4
        batch_size = 16

        model = keras.Sequential([
            keras.Input(shape=(input_dim,)),
            keras.layers.Dense(8, activation='relu'),
            HierarchicalRoutingLayer(output_dim=output_dim)
        ])

        model.compile(optimizer='adam', loss='categorical_crossentropy')

        x = np.random.randn(batch_size, input_dim).astype(np.float32)
        y = np.eye(output_dim)[np.random.choice(output_dim, batch_size)].astype(np.float32)

        # Check loss before training
        initial_loss = model.evaluate(x, y, verbose=0)

        # Train for a few steps
        model.fit(x, y, epochs=5, verbose=0)

        # Check loss decreased
        final_loss = model.evaluate(x, y, verbose=0)
        assert final_loss < initial_loss

        # Explicit check for gradients on the routing layer kernel
        routing_layer = model.layers[-1]
        assert routing_layer.kernel is not None

        # Simple gradient check
        with tf.GradientTape() as tape:
            preds = model(x)
            # FIX: Use a loss that varies with the distribution.
            # sum(probs) is always 1, so gradients would be 0.
            # sum(probs^2) varies (min at uniform, max at one-hot).
            loss = tf.reduce_mean(tf.square(preds))
        grads = tape.gradient(loss, routing_layer.trainable_weights)

        # Ensure gradients are being computed for kernel and bias
        assert len(grads) == 2
        assert np.any(grads[0].numpy() != 0), "Kernel gradients are zero (loss might be constant w.r.t weights)"
        assert np.any(grads[1].numpy() != 0), "Bias gradients are zero"

    def test_model_save_and_load(self):
        """Test full model saving and loading."""
        model = keras.Sequential([
            keras.layers.Dense(4, input_shape=(4,)),
            HierarchicalRoutingLayer(output_dim=3)
        ])

        x = np.random.randn(1, 4).astype(np.float32)
        y_ref = model.predict(x)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'model.keras')
            model.save(path)

            loaded_model = keras.models.load_model(path)
            y_new = loaded_model.predict(x)

            np.testing.assert_allclose(y_ref, y_new, atol=1e-6)
            assert isinstance(loaded_model.layers[-1], HierarchicalRoutingLayer)

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])