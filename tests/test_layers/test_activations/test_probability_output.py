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


# R-038 closure -- plan-2026-08-22T035419-a11304c8 / D-251.
# Keras `ops/nn.py:907` advises that a softmax over a size-1 axis always returns
# exactly 1.0. Every site in this module feeds that axis a size of 1 ON PURPOSE
# -- single class, single token, single head, single anchor, single cluster,
# minimum sequence length -- so the advisory describes the test's own input, not
# a defect. Suppressed HERE rather than in `pyproject.toml` so an ACCIDENTAL
# size-1 softmax anywhere else still fails under `error::UserWarning`.
pytestmark = [
    pytest.mark.filterwarnings(
        "ignore:You are using a softmax over axis:UserWarning"),
]


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

# ---------------------------------------------------------------------
# Mechanism oracles -- plan-2026-08-27T103353-60745fe0 / iter-2/step-9.
#
# The iter-2/step-8 mutation probe inserted `inputs = keras.ops.zeros_like(
# inputs)` at the top of `ProbabilityOutput.call`, so every strategy still
# returned a valid distribution of the right shape while the output became
# COMPLETELY input-independent. Exactly 1 of the 52 pre-existing tests fired
# --  `TestProbabilityOutputIntegration::test_gradient_flow` -- and it fired
# through the weakest oracle class in the probe's ranking, by accident of the
# mutation's shape rather than by testing the dispatcher's job. The probe's
# own summary rules this suite EFFECTIVELY BLIND and asks step 9 to treat it
# as one: a mutation that preserved d(out)/d(in) would have read fully blind.
#
# Two oracles below, neither of which needs a reference implementation of
# ProbabilityOutput itself:
#
#   1. Input-dependence, across all ten accepted `probability_type` spellings.
#      A dispatcher's output MUST depend on the input through the selected
#      strategy. This is the cheapest oracle in the probe's ranked list and it
#      kills the zeroed-input mutation on its own.
#   2. Dispatch identity: for the four deterministic, stateless logit
#      strategies, the dispatcher's output must equal what the named strategy
#      layer produces standalone -- and must NOT equal a plain softmax where
#      the strategy is not softmax. That catches a MIS-dispatch (routing to
#      the wrong strategy), which input-dependence alone would miss.
#
# The two routing strategies are excluded from oracle 2 because they own
# randomly-initialized projection weights, so a second instance is a
# different function; oracle 1 covers them.
# ---------------------------------------------------------------------

from dl_techniques.layers.activations.adaptive_softmax import AdaptiveTemperatureSoftmax
from dl_techniques.layers.activations.sparsemax import Sparsemax
from dl_techniques.layers.activations.thresh_max import ThreshMax

#: Every spelling `_SUPPORTED_TYPES` accepts. `output_dim` for the routing
#: strategies goes inside `type_config`, never as a direct kwarg.
_ALL_TYPES = [
    "softmax", "sparsemax", "threshmax", "thresh_max",
    "adaptive", "adaptive_softmax",
    "routing", "deterministic_routing",
    "hierarchical", "hierarchical_routing",
]

_ROUTING_TYPES = frozenset(
    {"routing", "deterministic_routing", "hierarchical", "hierarchical_routing"}
)


def _type_config_for(probability_type: str) -> dict:
    """`{'output_dim': 5}` for the routing strategies, `{}` otherwise.

    5 is a non-power-of-two on purpose: the hierarchical strategy's structural
    padding mask is an exact no-op at a power-of-two `output_dim`, so a test
    written at `output_dim=4` or `8` would silently stop exercising it.
    """
    return {"output_dim": 5} if probability_type in _ROUTING_TYPES else {}


def _two_different_inputs() -> tuple:
    """Two `(4, 16)` float32 batches, seed 11, scaled by 2.0.

    The scale pushes the logits far enough apart that every strategy -- even
    the flattest -- separates them by well over the assertion threshold.
    """
    rng = np.random.default_rng(11)
    a = (rng.standard_normal((4, 16)) * 2.0).astype("float32")
    b = (rng.standard_normal((4, 16)) * 2.0).astype("float32")
    return a, b


class TestProbabilityOutputDependsOnItsInput:

    @pytest.mark.parametrize("probability_type", _ALL_TYPES)
    def test_two_different_inputs_give_two_different_outputs(
            self, probability_type: str
    ) -> None:
        """Materially different inputs must give materially different outputs.

        ONE layer instance is used for both calls, so the routing strategies'
        random weights are held fixed and the only thing that varies is the
        input.

        Threshold 0.05. Derivation: the measured `max|f(a) - f(b)|` on this
        pair is 0.573528 for the flattest strategy (`softmax`) and up to
        1.000000 (`sparsemax`); across all ten spellings the minimum is
        0.573528. 0.05 is >11x below that minimum and ~6 orders of magnitude
        above float32 noise. The step-8 mutation drives this quantity to
        exactly 0.0 for every strategy.
        """
        layer = ProbabilityOutput(
            probability_type=probability_type,
            type_config=_type_config_for(probability_type),
        )
        a, b = _two_different_inputs()
        ya = ops.convert_to_numpy(layer(a))
        yb = ops.convert_to_numpy(layer(b))

        assert np.abs(ya - yb).max() > 0.05


class TestProbabilityOutputDispatchesToTheNamedStrategy:

    @staticmethod
    def _standalone(probability_type: str):
        """Build the strategy the name promises, directly and independently."""
        return {
            "softmax": lambda: None,
            "sparsemax": Sparsemax,
            "threshmax": ThreshMax,
            "thresh_max": ThreshMax,
            "adaptive": AdaptiveTemperatureSoftmax,
            "adaptive_softmax": AdaptiveTemperatureSoftmax,
        }[probability_type]()

    @pytest.mark.parametrize(
        "probability_type",
        ["softmax", "sparsemax", "threshmax", "thresh_max",
         "adaptive", "adaptive_softmax"],
    )
    def test_output_equals_the_named_strategy_computed_standalone(
            self, probability_type: str
    ) -> None:
        """The dispatcher reproduces the named strategy BIT-FOR-BIT.

        Tolerance atol=0, rtol=0 -- exact equality. Derivation: this is not a
        numerical claim but a routing claim. The dispatcher forwards the same
        tensor to the same op sequence, so the results are identical bit
        patterns; measured `max|dispatcher - standalone|` is exactly 0.000e+00
        for all six spellings. A non-zero tolerance here would accept a
        DIFFERENT strategy that happens to be numerically close.

        These six strategies are deterministic and carry no randomly
        initialized weights (`ThreshMax`'s `slope_weight` defaults to the
        constant 10.0 and `trainable_slope=False`), which is what makes a
        standalone instance a valid oracle for them.
        """
        a, _ = _two_different_inputs()
        dispatched = ops.convert_to_numpy(
            ProbabilityOutput(probability_type=probability_type)(a)
        )

        if probability_type == "softmax":
            expected = ops.convert_to_numpy(
                ops.softmax(ops.convert_to_tensor(a), axis=-1)
            )
        else:
            expected = ops.convert_to_numpy(self._standalone(probability_type)(a))

        np.testing.assert_array_equal(dispatched, expected)

    @pytest.mark.parametrize(
        "probability_type",
        ["sparsemax", "threshmax", "thresh_max", "adaptive", "adaptive_softmax"],
    )
    def test_a_non_softmax_strategy_is_not_a_plain_softmax(
            self, probability_type: str
    ) -> None:
        """Every non-softmax strategy must differ from a plain softmax.

        The negative half of the dispatch claim: without it, a dispatcher that
        collapsed every key onto `softmax` would satisfy the equality arm for
        `softmax` and be unfalsifiable everywhere else.

        Threshold 0.05. Derivation: the measured `max|strategy - softmax|` on
        this input is 0.624859 (`sparsemax`), 0.127702 (`threshmax`) and
        0.694678 (`adaptive`). 0.05 is 2.6x below the SMALLEST of those --
        `threshmax`, which is the closest of the three to a plain softmax by
        construction, since it is a softmax with a confidence gate.
        """
        a, _ = _two_different_inputs()
        dispatched = ops.convert_to_numpy(
            ProbabilityOutput(probability_type=probability_type)(a)
        )
        plain = ops.convert_to_numpy(ops.softmax(ops.convert_to_tensor(a), axis=-1))

        assert np.abs(dispatched - plain).max() > 0.05

    def test_sparsemax_alone_produces_exact_zeros_on_a_peaked_input(self) -> None:
        """A behavioural discriminator that does not read any strategy's code.

        `sparsemax` is the Euclidean projection onto the simplex, so on a
        peaked input it assigns EXACTLY zero mass outside the support;
        `softmax` is dense and can never return an exact zero for a finite
        logit. Measured on the input below: sparsemax gives 15.0 exact zeros
        per row of 16, softmax gives 0.0. Exact equality with 0.0, no
        tolerance -- the claim is about which values are identically zero.
        """
        peaked = np.zeros((4, 16), dtype="float32")
        peaked[:, 0] = 8.0

        sparse = ops.convert_to_numpy(
            ProbabilityOutput(probability_type="sparsemax")(peaked)
        )
        dense = ops.convert_to_numpy(
            ProbabilityOutput(probability_type="softmax")(peaked)
        )

        assert (sparse == 0.0).sum(axis=-1).min() >= 14
        assert not np.any(dense == 0.0)
