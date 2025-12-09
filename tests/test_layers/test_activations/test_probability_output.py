"""
Tests for the ProbabilityOutput unified probability layer.

This module provides comprehensive pytest-based tests for the ProbabilityOutput
layer, covering instantiation, forward pass, serialization, and edge cases.
"""

import tempfile
import os

import pytest
import numpy as np
import keras
from keras import ops

from dl_techniques.layers.activations.probability_output import ProbabilityOutput


class TestProbabilityOutputInstantiation:
    """Tests for layer instantiation and configuration validation."""

    def test_default_instantiation(self):
        """Test layer instantiates with default softmax type."""
        layer = ProbabilityOutput()
        assert layer.probability_type == "softmax"
        assert layer.type_config == {}

    @pytest.mark.parametrize("prob_type", [
        "softmax",
        "sparsemax",
        "threshmax",
        "thresh_max",
        "adaptive",
        "adaptive_softmax",
    ])
    def test_logit_based_types_instantiation(self, prob_type: str):
        """Test instantiation of logit-based probability types."""
        layer = ProbabilityOutput(probability_type=prob_type)
        assert layer.probability_type == prob_type.lower()
        assert layer.strategy_layer is not None

    @pytest.mark.parametrize("prob_type", [
        "routing",
        "deterministic_routing",
    ])
    def test_routing_types_instantiation(self, prob_type: str):
        """Test instantiation of routing-based probability types."""
        layer = ProbabilityOutput(
            probability_type=prob_type,
            type_config={"output_dim": 10}
        )
        assert layer.probability_type == prob_type.lower()
        assert layer.strategy_layer is not None

    @pytest.mark.parametrize("prob_type", [
        "hierarchical",
        "hierarchical_routing",
    ])
    def test_hierarchical_types_instantiation(self, prob_type: str):
        """Test instantiation of hierarchical routing types."""
        layer = ProbabilityOutput(
            probability_type=prob_type,
            type_config={"output_dim": 10}
        )
        assert layer.probability_type == prob_type.lower()
        assert layer.strategy_layer is not None

    def test_hierarchical_requires_output_dim(self):
        """Test that hierarchical type requires output_dim in config."""
        with pytest.raises(ValueError, match="requires 'output_dim'"):
            ProbabilityOutput(probability_type="hierarchical")

    def test_hierarchical_routing_requires_output_dim(self):
        """Test that hierarchical_routing type requires output_dim in config."""
        with pytest.raises(ValueError, match="requires 'output_dim'"):
            ProbabilityOutput(probability_type="hierarchical_routing")

    def test_invalid_probability_type(self):
        """Test that invalid probability type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown probability_type"):
            ProbabilityOutput(probability_type="invalid_type")

    def test_type_config_preserved(self):
        """Test that type_config is correctly stored."""
        config = {"axis": -2, "slope": 5.0}
        layer = ProbabilityOutput(
            probability_type="threshmax",
            type_config=config
        )
        assert layer.type_config == config

    def test_type_config_returns_copy(self):
        """Test that type_config property returns a copy."""
        config = {"axis": -1}
        layer = ProbabilityOutput(
            probability_type="softmax",
            type_config=config
        )
        returned_config = layer.type_config
        returned_config["new_key"] = "new_value"
        assert "new_key" not in layer.type_config

    def test_case_insensitive_type(self):
        """Test that probability_type is case-insensitive."""
        layer = ProbabilityOutput(probability_type="SOFTMAX")
        assert layer.probability_type == "softmax"

        layer2 = ProbabilityOutput(probability_type="SpArSeMaX")
        assert layer2.probability_type == "sparsemax"


class TestProbabilityOutputForwardPass:
    """Tests for forward pass computation."""

    @pytest.fixture
    def sample_logits(self) -> np.ndarray:
        """Generate sample logits for testing."""
        return np.random.randn(8, 10).astype(np.float32)

    @pytest.fixture
    def sample_features(self) -> np.ndarray:
        """Generate sample features for routing-based tests."""
        return np.random.randn(8, 64).astype(np.float32)

    @pytest.fixture
    def sample_3d_logits(self) -> np.ndarray:
        """Generate 3D sample logits for sequence testing."""
        return np.random.randn(4, 16, 10).astype(np.float32)

    def test_softmax_forward_pass(self, sample_logits: np.ndarray):
        """Test softmax forward pass produces valid probabilities."""
        layer = ProbabilityOutput(probability_type="softmax")
        output = layer(sample_logits)

        assert output.shape == sample_logits.shape
        # Check probabilities sum to 1
        sums = keras.ops.sum(output, axis=-1)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(sums),
            np.ones(sample_logits.shape[0]),
            rtol=1e-5, atol=1e-5,
            err_msg="Softmax outputs should sum to 1"
        )
        # Check all values are non-negative
        assert keras.ops.all(output >= 0)

    def test_sparsemax_forward_pass(self, sample_logits: np.ndarray):
        """Test sparsemax forward pass produces valid sparse probabilities."""
        layer = ProbabilityOutput(probability_type="sparsemax")
        output = layer(sample_logits)

        assert output.shape == sample_logits.shape
        # Check probabilities sum to 1
        sums = keras.ops.sum(output, axis=-1)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(sums),
            np.ones(sample_logits.shape[0]),
            rtol=1e-5, atol=1e-5,
            err_msg="Sparsemax outputs should sum to 1"
        )
        # Check all values are non-negative
        assert keras.ops.all(output >= 0)

    def test_threshmax_forward_pass(self, sample_logits: np.ndarray):
        """Test threshmax forward pass produces valid probabilities."""
        layer = ProbabilityOutput(
            probability_type="threshmax",
            type_config={"slope": 10.0}
        )
        output = layer(sample_logits)

        assert output.shape == sample_logits.shape
        # Check all values are non-negative
        assert keras.ops.all(output >= 0)

    def test_adaptive_forward_pass(self, sample_logits: np.ndarray):
        """Test adaptive softmax forward pass."""
        layer = ProbabilityOutput(
            probability_type="adaptive",
            type_config={"min_temp": 0.1, "max_temp": 1.0}
        )
        output = layer(sample_logits)

        assert output.shape == sample_logits.shape
        # Check probabilities sum to 1
        sums = keras.ops.sum(output, axis=-1)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(sums),
            np.ones(sample_logits.shape[0]),
            rtol=1e-5, atol=1e-5,
            err_msg="Adaptive softmax outputs should sum to 1"
        )

    def test_routing_forward_pass(self, sample_features: np.ndarray):
        """Test routing forward pass with features input."""
        output_dim = 10
        layer = ProbabilityOutput(
            probability_type="routing",
            type_config={"output_dim": output_dim}
        )
        output = layer(sample_features)

        assert output.shape == (sample_features.shape[0], output_dim)

    def test_hierarchical_forward_pass(self, sample_features: np.ndarray):
        """Test hierarchical routing forward pass."""
        output_dim = 10
        layer = ProbabilityOutput(
            probability_type="hierarchical",
            type_config={"output_dim": output_dim}
        )
        output = layer(sample_features)

        assert output.shape == (sample_features.shape[0], output_dim)

    def test_3d_input_softmax(self, sample_3d_logits: np.ndarray):
        """Test softmax handles 3D input correctly."""
        layer = ProbabilityOutput(probability_type="softmax")
        output = layer(sample_3d_logits)

        assert output.shape == sample_3d_logits.shape

    def test_training_mode_passed(self, sample_logits: np.ndarray):
        """Test that training mode is passed to strategy layer."""
        layer = ProbabilityOutput(
            probability_type="threshmax",
            type_config={"trainable_slope": True}
        )

        output_train = layer(sample_logits, training=True)
        output_eval = layer(sample_logits, training=False)

        assert output_train.shape == sample_logits.shape
        assert output_eval.shape == sample_logits.shape

    def test_softmax_with_mask(self, sample_logits: np.ndarray):
        """Test softmax forward pass with mask."""
        layer = ProbabilityOutput(probability_type="softmax")
        mask = np.ones((8, 10), dtype=np.float32)
        mask[:, -2:] = 0  # Mask last 2 positions

        output = layer(sample_logits, mask=mask)
        assert output.shape == sample_logits.shape


class TestProbabilityOutputBuild:
    """Tests for layer build behavior."""

    def test_build_creates_strategy_weights(self):
        """Test that build properly builds the strategy layer."""
        layer = ProbabilityOutput(
            probability_type="threshmax",
            type_config={"trainable_slope": True}
        )
        layer.build((None, 10))

        assert layer.built
        assert layer.strategy_layer.built

    def test_build_idempotent(self):
        """Test that calling build multiple times is safe."""
        layer = ProbabilityOutput(probability_type="softmax")
        layer.build((None, 10))
        layer.build((None, 10))

        assert layer.built


class TestProbabilityOutputOutputShape:
    """Tests for compute_output_shape."""

    def test_softmax_output_shape(self):
        """Test softmax preserves input shape."""
        layer = ProbabilityOutput(probability_type="softmax")
        input_shape = (None, 10)
        output_shape = layer.compute_output_shape(input_shape)

        assert output_shape == input_shape

    def test_sparsemax_output_shape(self):
        """Test sparsemax preserves input shape."""
        layer = ProbabilityOutput(probability_type="sparsemax")
        input_shape = (None, 20)
        output_shape = layer.compute_output_shape(input_shape)

        assert output_shape == input_shape

    def test_routing_output_shape(self):
        """Test routing changes output dimension."""
        output_dim = 15
        layer = ProbabilityOutput(
            probability_type="routing",
            type_config={"output_dim": output_dim}
        )
        input_shape = (None, 64)
        output_shape = layer.compute_output_shape(input_shape)

        assert output_shape[-1] == output_dim

    def test_3d_output_shape(self):
        """Test output shape computation for 3D inputs."""
        layer = ProbabilityOutput(probability_type="softmax")
        input_shape = (None, 16, 10)
        output_shape = layer.compute_output_shape(input_shape)

        assert output_shape == input_shape


class TestProbabilityOutputSerialization:
    """Tests for serialization and deserialization."""

    @pytest.fixture
    def sample_input(self) -> np.ndarray:
        """Generate sample input for serialization tests."""
        return np.random.randn(4, 10).astype(np.float32)

    def test_get_config_softmax(self):
        """Test get_config returns complete configuration for softmax."""
        layer = ProbabilityOutput(
            probability_type="softmax",
            type_config={"axis": -1},
            name="test_prob"
        )
        config = layer.get_config()

        assert config["probability_type"] == "softmax"
        assert config["type_config"] == {"axis": -1}
        assert config["name"] == "test_prob"

    def test_get_config_threshmax(self):
        """Test get_config for threshmax with custom config."""
        type_config = {"slope": 15.0, "trainable_slope": True}
        layer = ProbabilityOutput(
            probability_type="threshmax",
            type_config=type_config
        )
        config = layer.get_config()

        assert config["probability_type"] == "threshmax"
        assert config["type_config"] == type_config

    def test_from_config_reconstruction(self):
        """Test layer can be reconstructed from config."""
        original = ProbabilityOutput(
            probability_type="adaptive",
            type_config={"min_temp": 0.05, "max_temp": 2.0}
        )
        config = original.get_config()

        reconstructed = ProbabilityOutput.from_config(config)

        assert reconstructed.probability_type == original.probability_type
        assert reconstructed.type_config == original.type_config

    @pytest.mark.parametrize("prob_type,type_config", [
        ("softmax", {"axis": -1}),
        ("sparsemax", {}),
        ("threshmax", {"slope": 10.0}),
        ("adaptive", {"min_temp": 0.1, "max_temp": 1.0}),
    ])
    def test_serialization_cycle_logit_based(
            self,
            prob_type: str,
            type_config: dict,
            sample_input: np.ndarray
    ):
        """Test full save/load cycle for logit-based strategies."""
        layer = ProbabilityOutput(
            probability_type=prob_type,
            type_config=type_config
        )

        # Build model for serialization
        inputs = keras.Input(shape=(10,))
        outputs = layer(inputs)
        model = keras.Model(inputs, outputs)

        # Get output before saving
        original_output = model(sample_input)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "test_model.keras")
            model.save(model_path)

            loaded_model = keras.models.load_model(model_path)
            loaded_output = loaded_model(sample_input)

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(original_output),
            keras.ops.convert_to_numpy(loaded_output),
            rtol=1e-6, atol=1e-6,
            err_msg=f"Outputs should match after serialization for {prob_type}"
        )

    def test_serialization_cycle_routing(self):
        """Test full save/load cycle for routing strategy."""
        sample_features = np.random.randn(4, 32).astype(np.float32)
        output_dim = 10

        layer = ProbabilityOutput(
            probability_type="routing",
            type_config={"output_dim": output_dim}
        )

        inputs = keras.Input(shape=(32,))
        outputs = layer(inputs)
        model = keras.Model(inputs, outputs)

        original_output = model(sample_features)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "test_routing.keras")
            model.save(model_path)

            loaded_model = keras.models.load_model(model_path)
            loaded_output = loaded_model(sample_features)

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(original_output),
            keras.ops.convert_to_numpy(loaded_output),
            rtol=1e-6, atol=1e-6,
            err_msg="Routing outputs should match after serialization"
        )

    def test_serialization_cycle_hierarchical(self):
        """Test full save/load cycle for hierarchical strategy."""
        sample_features = np.random.randn(4, 32).astype(np.float32)
        output_dim = 10

        layer = ProbabilityOutput(
            probability_type="hierarchical",
            type_config={"output_dim": output_dim}
        )

        inputs = keras.Input(shape=(32,))
        outputs = layer(inputs)
        model = keras.Model(inputs, outputs)

        original_output = model(sample_features)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "test_hierarchical.keras")
            model.save(model_path)

            loaded_model = keras.models.load_model(model_path)
            loaded_output = loaded_model(sample_features)

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(original_output),
            keras.ops.convert_to_numpy(loaded_output),
            rtol=1e-6, atol=1e-6,
            err_msg="Hierarchical outputs should match after serialization"
        )


class TestProbabilityOutputEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_class_softmax(self):
        """Test softmax with single class input."""
        layer = ProbabilityOutput(probability_type="softmax")
        single_class_input = np.random.randn(4, 1).astype(np.float32)
        output = layer(single_class_input)

        assert output.shape == single_class_input.shape
        # Single class should always be 1.0
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(output),
            np.ones_like(single_class_input),
            rtol=1e-5, atol=1e-5
        )

    def test_large_logits(self):
        """Test numerical stability with large logit values."""
        layer = ProbabilityOutput(probability_type="softmax")
        large_logits = np.array([[1000.0, 1.0, 0.0]], dtype=np.float32)
        output = layer(large_logits)

        # Should not produce NaN or Inf
        assert not np.any(np.isnan(keras.ops.convert_to_numpy(output)))
        assert not np.any(np.isinf(keras.ops.convert_to_numpy(output)))

    def test_negative_logits(self):
        """Test with all negative logits."""
        layer = ProbabilityOutput(probability_type="softmax")
        negative_logits = np.array([[-10.0, -5.0, -1.0]], dtype=np.float32)
        output = layer(negative_logits)

        sums = keras.ops.sum(output, axis=-1)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(sums),
            np.ones(1),
            rtol=1e-5, atol=1e-5
        )

    def test_batch_size_one(self):
        """Test with batch size of 1."""
        layer = ProbabilityOutput(probability_type="sparsemax")
        single_batch = np.random.randn(1, 10).astype(np.float32)
        output = layer(single_batch)

        assert output.shape == (1, 10)

    def test_many_classes(self):
        """Test with large number of classes."""
        layer = ProbabilityOutput(probability_type="softmax")
        many_classes = np.random.randn(4, 10000).astype(np.float32)
        output = layer(many_classes)

        assert output.shape == (4, 10000)
        sums = keras.ops.sum(output, axis=-1)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(sums),
            np.ones(4),
            rtol=1e-4, atol=1e-4
        )

    def test_empty_type_config(self):
        """Test that None type_config is handled as empty dict."""
        layer = ProbabilityOutput(
            probability_type="softmax",
            type_config=None
        )
        assert layer.type_config == {}

    def test_custom_axis(self):
        """Test softmax with non-default axis."""
        layer = ProbabilityOutput(
            probability_type="softmax",
            type_config={"axis": 1}
        )
        input_3d = np.random.randn(4, 10, 5).astype(np.float32)
        output = layer(input_3d)

        # Sum along axis 1 should be 1
        sums = keras.ops.sum(output, axis=1)
        expected = np.ones((4, 5), dtype=np.float32)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(sums),
            expected,
            rtol=1e-5, atol=1e-5
        )


class TestProbabilityOutputIntegration:
    """Integration tests with models."""

    def test_in_sequential_model(self):
        """Test layer works in Sequential model."""
        model = keras.Sequential([
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(10),
            ProbabilityOutput(probability_type="softmax"),
        ])

        sample_input = np.random.randn(8, 32).astype(np.float32)
        output = model(sample_input)

        assert output.shape == (8, 10)

    def test_in_functional_model(self):
        """Test layer works in Functional API model."""
        inputs = keras.Input(shape=(32,))
        x = keras.layers.Dense(64, activation="relu")(inputs)
        logits = keras.layers.Dense(10)(x)
        outputs = ProbabilityOutput(probability_type="sparsemax")(logits)

        model = keras.Model(inputs, outputs)
        sample_input = np.random.randn(8, 32).astype(np.float32)
        output = model(sample_input)

        assert output.shape == (8, 10)

    def test_multiple_probability_outputs(self):
        """Test model with multiple ProbabilityOutput layers."""
        inputs = keras.Input(shape=(32,))
        x = keras.layers.Dense(64, activation="relu")(inputs)

        logits1 = keras.layers.Dense(10)(x)
        logits2 = keras.layers.Dense(5)(x)

        out1 = ProbabilityOutput(probability_type="softmax", name="prob_1")(logits1)
        out2 = ProbabilityOutput(probability_type="sparsemax", name="prob_2")(logits2)

        model = keras.Model(inputs, [out1, out2])
        sample_input = np.random.randn(8, 32).astype(np.float32)
        outputs = model(sample_input)

        assert outputs[0].shape == (8, 10)
        assert outputs[1].shape == (8, 5)

    def test_gradient_flow(self):
        """Test that gradients flow through the layer."""
        import tensorflow as tf

        layer = ProbabilityOutput(probability_type="softmax")
        inputs = tf.Variable(np.random.randn(4, 10).astype(np.float32))

        with tf.GradientTape() as tape:
            outputs = layer(inputs)
            loss = tf.reduce_mean(outputs)

        gradients = tape.gradient(loss, inputs)

        assert gradients is not None
        assert gradients.shape == inputs.shape

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])