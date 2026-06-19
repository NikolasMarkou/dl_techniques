import pytest
import tempfile
import os
import numpy as np
import keras
from typing import Any, Dict
import tensorflow as tf

from dl_techniques.layers.ffn.squared_relu_ffn import SquaredReLUFFN
from dl_techniques.layers.ffn.factory import create_ffn_layer


class TestSquaredReLUFFN:
    """Comprehensive test suite for the SquaredReLUFFN layer."""

    @pytest.fixture
    def layer_config(self) -> Dict[str, Any]:
        """Standard configuration for testing."""
        return {
            'hidden_dim': 128,
            'output_dim': 64,
            'dropout_rate': 0.1,
            'use_bias': True,
            'kernel_initializer': 'glorot_uniform',
            'bias_initializer': 'zeros',
        }

    @pytest.fixture
    def sample_input_2d(self) -> keras.KerasTensor:
        """Sample 2D input."""
        return keras.random.normal(shape=(4, 32))

    @pytest.fixture
    def sample_input_3d(self) -> keras.KerasTensor:
        """Sample 3D input (sequence data)."""
        return keras.random.normal(shape=(2, 16, 32))

    def test_initialization(self, layer_config: Dict[str, Any]) -> None:
        """Test layer initialization stores all params (catches factory param-drop)."""
        layer = SquaredReLUFFN(**layer_config)

        assert layer.hidden_dim == layer_config['hidden_dim']
        assert layer.output_dim == layer_config['output_dim']
        assert layer.dropout_rate == layer_config['dropout_rate']
        assert layer.use_bias == layer_config['use_bias']

        # Not built yet; Dense weights created on build()
        assert not layer.built

        # Sub-layers created in __init__
        assert isinstance(layer.fc1, keras.layers.Dense)
        assert isinstance(layer.fc2, keras.layers.Dense)
        assert isinstance(layer.dropout, keras.layers.Dropout)
        assert layer.fc1.units == layer_config['hidden_dim']
        assert layer.fc2.units == layer_config['output_dim']
        assert layer.dropout.rate == layer_config['dropout_rate']

        # No configurable activation parameter on this layer (squared-relu is fixed)
        assert not hasattr(layer, 'activation_fn')

    @pytest.mark.parametrize("sample_input,leading", [
        ("sample_input_2d", (4,)),
        ("sample_input_3d", (2, 16)),
    ])
    def test_forward_pass(self, layer_config: Dict[str, Any],
                          sample_input: str, leading, request) -> None:
        """Test forward pass + build on 2D / 3D inputs."""
        inputs = request.getfixturevalue(sample_input)
        layer = SquaredReLUFFN(**layer_config)

        output = layer(inputs)

        assert layer.built

        expected_shape = (*leading, layer_config['output_dim'])
        assert output.shape == expected_shape
        assert output.shape[-1] == layer_config['output_dim']

        output_numpy = keras.ops.convert_to_numpy(output)
        assert np.isfinite(output_numpy).all()

    def test_activation_math(self, sample_input_2d: keras.KerasTensor) -> None:
        """Prove the fixed squared-ReLU is applied: fc2(square(relu(fc1(x))))."""
        layer = SquaredReLUFFN(hidden_dim=128, output_dim=64, dropout_rate=0.0)
        # Build by calling once.
        output = layer(sample_input_2d)

        # Manually reproduce using the layer's own sublayers.
        h = layer.fc1(sample_input_2d)
        h = keras.ops.square(keras.ops.relu(h))
        manual = layer.fc2(h)

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(output),
            keras.ops.convert_to_numpy(manual),
            rtol=1e-6, atol=1e-6,
            err_msg="Output does not match fc2(square(relu(fc1(x))))"
        )

    def test_compute_output_shape(self, layer_config: Dict[str, Any]) -> None:
        """Test compute_output_shape transforms only the last dim."""
        layer = SquaredReLUFFN(**layer_config)

        test_shapes = [
            (None, 32),
            (None, 16, 32),
            (None, 8, 8, 32),
        ]
        for input_shape in test_shapes:
            output_shape = layer.compute_output_shape(input_shape)
            assert output_shape[-1] == layer_config['output_dim']
            assert output_shape[:-1] == input_shape[:-1]

    def test_serialization_cycle(self, layer_config: Dict[str, Any],
                                 sample_input_3d: keras.KerasTensor) -> None:
        """CRITICAL: full .keras serialization cycle with prediction comparison."""
        inputs = keras.Input(shape=sample_input_3d.shape[1:])
        outputs = SquaredReLUFFN(**layer_config)(inputs)
        model = keras.Model(inputs, outputs)

        original_prediction = model(sample_input_3d)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_model.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_prediction = loaded_model(sample_input_3d)

            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_prediction),
                keras.ops.convert_to_numpy(loaded_prediction),
                rtol=1e-6, atol=1e-6,
                err_msg="Predictions differ after serialization"
            )

    def test_config_completeness(self, layer_config: Dict[str, Any]) -> None:
        """Test that get_config contains every __init__ param."""
        layer = SquaredReLUFFN(**layer_config)
        config = layer.get_config()

        required_keys = {
            'hidden_dim', 'output_dim', 'dropout_rate', 'use_bias',
            'kernel_initializer', 'bias_initializer',
            'kernel_regularizer', 'bias_regularizer'
        }
        for key in required_keys:
            assert key in config, f"Missing {key} in get_config()"

        assert config['hidden_dim'] == layer_config['hidden_dim']
        assert config['output_dim'] == layer_config['output_dim']
        assert config['dropout_rate'] == layer_config['dropout_rate']
        assert config['use_bias'] == layer_config['use_bias']

        # No activation param should leak into config (fixed non-linearity)
        assert 'activation' not in config

    def test_gradients_flow(self, layer_config: Dict[str, Any],
                            sample_input_3d: keras.KerasTensor) -> None:
        """Test that gradients flow to all trainable weights and are non-zero."""
        layer = SquaredReLUFFN(**layer_config)

        with tf.GradientTape() as tape:
            tape.watch(sample_input_3d)
            output = layer(sample_input_3d)
            loss = keras.ops.mean(keras.ops.square(output))

        input_gradients = tape.gradient(loss, sample_input_3d)
        assert input_gradients is not None
        assert input_gradients.shape == sample_input_3d.shape

        with tf.GradientTape() as weight_tape:
            output = layer(sample_input_3d)
            loss = keras.ops.mean(keras.ops.square(output))

        weight_gradients = weight_tape.gradient(loss, layer.trainable_variables)

        assert len(weight_gradients) > 0
        for grad in weight_gradients:
            assert grad is not None
            grad_numpy = keras.ops.convert_to_numpy(grad)
            assert not np.allclose(grad_numpy, 0), "Gradient is zero - possible dead weight"

    def test_edge_cases(self) -> None:
        """Test error conditions."""
        with pytest.raises(ValueError, match="hidden_dim must be positive"):
            SquaredReLUFFN(hidden_dim=0, output_dim=64)

        with pytest.raises(ValueError, match="output_dim must be positive"):
            SquaredReLUFFN(hidden_dim=64, output_dim=-1)

        with pytest.raises(ValueError, match="dropout_rate must be in"):
            SquaredReLUFFN(hidden_dim=64, output_dim=64, dropout_rate=1.1)

    def test_factory_creation(self, sample_input_2d: keras.KerasTensor) -> None:
        """Test creation through the FFN factory (catches param-drop trap)."""
        layer = create_ffn_layer('squared_relu', hidden_dim=128, output_dim=64)

        assert isinstance(layer, SquaredReLUFFN)
        assert layer.hidden_dim == 128
        assert layer.output_dim == 64

        output = layer(sample_input_2d)
        assert output.shape == (4, 64)
        output_numpy = keras.ops.convert_to_numpy(output)
        assert np.isfinite(output_numpy).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
