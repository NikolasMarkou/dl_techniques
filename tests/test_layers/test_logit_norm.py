import sys
from pathlib import Path
import pytest
import tensorflow as tf

# Add the src directory to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / "src"))

from dl_techniques.layers.logit_norm import CoupledLogitNorm, CoupledMultiLabelHead


@pytest.fixture
def default_inputs():
    """Fixture for creating default test inputs."""
    return tf.constant([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]], dtype=tf.float32)


@pytest.fixture
def layer_configs():
    """Fixture for different layer configurations to test."""
    return [
        {"constant": 1.0, "coupling_strength": 1.0},
        {"constant": 2.0, "coupling_strength": 0.5},
        {"constant": 0.5, "coupling_strength": 2.0}
    ]


class TestCoupledLogitNorm:
    """Test suite for CoupledLogitNorm layer."""

    def test_initialization(self):
        """Test layer initialization with valid parameters."""
        layer = CoupledLogitNorm(constant=1.0, coupling_strength=1.0)
        assert layer.constant == 1.0
        assert layer.coupling_strength == 1.0
        assert layer.axis == -1
        assert layer.epsilon == 1e-7

    @pytest.mark.parametrize("invalid_config", [
        {"constant": -1.0},
        {"constant": 0.0},
        {"coupling_strength": -1.0},
        {"coupling_strength": 0.0},
        {"epsilon": -1e-7}
    ])
    def test_invalid_initialization(self, invalid_config):
        """Test layer initialization with invalid parameters."""
        with pytest.raises(ValueError):
            CoupledLogitNorm(**invalid_config)

    def test_output_shape(self, default_inputs):
        """Test output shape matches input shape."""
        layer = CoupledLogitNorm()
        output, norm = layer(default_inputs)
        assert output.shape == default_inputs.shape
        assert norm.shape == (2, 1)  # Batch size x 1 for norm

    @pytest.mark.parametrize("config", [
        {"constant": 1.0, "coupling_strength": 1.0},
        {"constant": 2.0, "coupling_strength": 0.5},
        {"constant": 0.5, "coupling_strength": 2.0}
    ])
    def test_normalization(self, default_inputs, config):
        """Test normalization behavior with different configurations."""
        layer = CoupledLogitNorm(**config)
        output, norm = layer(default_inputs)

        # Test that outputs maintain relative proportions
        input_ratios = default_inputs / tf.reduce_max(tf.abs(default_inputs), axis=-1, keepdims=True)
        output_ratios = output / tf.reduce_max(tf.abs(output), axis=-1, keepdims=True)
        tf.debugging.assert_near(input_ratios, output_ratios, rtol=1e-5)

        # Test that norm calculation is correct
        x_squared = tf.square(default_inputs)
        expected_norm = tf.reduce_sum(x_squared, axis=-1, keepdims=True)
        expected_norm = tf.pow(expected_norm + layer.epsilon, layer.coupling_strength / 2.0)
        tf.debugging.assert_near(norm, expected_norm, rtol=1e-5)

        # Test that the output scale is correct
        scaled_inputs = default_inputs / (norm * layer.constant)
        tf.debugging.assert_near(output, scaled_inputs, rtol=1e-5)

    def test_coupling_effect(self):
        """Test that coupling strength affects normalization."""
        inputs = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)

        # Compare outputs with different coupling strengths
        layer1 = CoupledLogitNorm(coupling_strength=1.0)
        layer2 = CoupledLogitNorm(coupling_strength=2.0)

        out1, _ = layer1(inputs)
        out2, _ = layer2(inputs)

        # Outputs should be different with different coupling strengths
        assert not tf.reduce_all(tf.abs(out1 - out2) < 1e-5)


class TestCoupledMultiLabelHead:
    """Test suite for CoupledMultiLabelHead layer."""

    def test_initialization(self):
        """Test layer initialization."""
        head = CoupledMultiLabelHead(constant=2.0, coupling_strength=0.5)
        assert isinstance(head.logit_norm, CoupledLogitNorm)
        assert head.logit_norm.constant == 2.0
        assert head.logit_norm.coupling_strength == 0.5

    def test_output_range(self, default_inputs):
        """Test that outputs are properly bounded by sigmoid."""
        head = CoupledMultiLabelHead()
        outputs = head(default_inputs)

        # Check outputs are in [0, 1]
        assert tf.reduce_all(outputs >= 0.0)
        assert tf.reduce_all(outputs <= 1.0)

    def test_gradient_flow(self, default_inputs):
        """Test gradient flow through the head."""
        head = CoupledMultiLabelHead()

        with tf.GradientTape() as tape:
            inputs = tf.Variable(default_inputs)
            outputs = head(inputs)
            loss = tf.reduce_mean(outputs)

        gradients = tape.gradient(loss, inputs)
        assert gradients is not None
        assert not tf.reduce_any(tf.math.is_nan(gradients))


@pytest.mark.integration
def test_integration():
    """Integration test with a simple model."""
    inputs = tf.keras.Input(shape=(3,))
    outputs = CoupledMultiLabelHead()(inputs)
    model = tf.keras.Model(inputs, outputs)

    # Test forward pass
    batch = tf.random.normal((4, 3))
    predictions = model(batch)

    assert predictions.shape == (4, 3)
    assert tf.reduce_all(predictions >= 0.0)
    assert tf.reduce_all(predictions <= 1.0)
