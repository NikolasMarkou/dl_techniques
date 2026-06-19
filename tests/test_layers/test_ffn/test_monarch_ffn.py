import pytest
import tempfile
import os
import numpy as np
import keras
from typing import Any, Dict
import tensorflow as tf

from dl_techniques.layers.ffn.monarch_ffn import MonarchFFN
from dl_techniques.layers.ffn.factory import create_ffn_layer


class TestMonarchFFN:
    """Comprehensive test suite for the MonarchFFN layer."""

    @pytest.fixture
    def layer_config(self) -> Dict[str, Any]:
        """Standard (square) configuration for testing."""
        return {
            'hidden_dim': 64,
            'output_dim': 64,
            'nblocks': 4,
            'activation': 'gelu',
            'dropout_rate': 0.1,
            'use_bias': True,
            'kernel_initializer': 'glorot_uniform',
            'bias_initializer': 'zeros',
        }

    @pytest.fixture
    def nonsquare_config(self) -> Dict[str, Any]:
        """Non-square configuration (hidden_dim != output_dim)."""
        return {
            'hidden_dim': 128,
            'output_dim': 64,
            'nblocks': 4,
            'activation': 'gelu',
            'dropout_rate': 0.0,
        }

    @pytest.fixture
    def sample_input_2d(self) -> keras.KerasTensor:
        """Sample 2D input (input_dim=32, divisible by nblocks=4)."""
        return keras.random.normal(shape=(4, 32))

    @pytest.fixture
    def sample_input_3d(self) -> keras.KerasTensor:
        """Sample 3D input (sequence data)."""
        return keras.random.normal(shape=(2, 16, 32))

    def test_initialization(self, layer_config: Dict[str, Any]) -> None:
        """Test layer initialization stores all params (catches factory param-drop)."""
        layer = MonarchFFN(**layer_config)

        assert layer.hidden_dim == layer_config['hidden_dim']
        assert layer.output_dim == layer_config['output_dim']
        assert layer.nblocks == layer_config['nblocks']
        assert layer.dropout_rate == layer_config['dropout_rate']
        assert layer.use_bias == layer_config['use_bias']

        # Not built yet; weights created in build()
        assert not layer.built
        assert layer.expand_l is None
        assert layer.contract_r is None

        # Dropout sub-layer created in __init__
        assert isinstance(layer.dropout, keras.layers.Dropout)
        assert layer.dropout.rate == layer_config['dropout_rate']
        assert layer.activation is not None

    @pytest.mark.parametrize("sample_input", ["sample_input_2d", "sample_input_3d"])
    def test_forward_pass(self, layer_config: Dict[str, Any],
                          sample_input: str, request) -> None:
        """Test forward pass + build for square config on 2D / 3D inputs."""
        inputs = request.getfixturevalue(sample_input)
        layer = MonarchFFN(**layer_config)

        output = layer(inputs)

        assert layer.built

        expected_shape = list(inputs.shape)
        expected_shape[-1] = layer_config['output_dim']
        assert output.shape == tuple(expected_shape)
        assert output.shape[-1] == layer_config['output_dim']

        # Monarch weights were created
        assert layer.expand_l is not None
        assert layer.expand_r is not None
        assert layer.contract_l is not None
        assert layer.contract_r is not None

        output_numpy = keras.ops.convert_to_numpy(output)
        assert np.isfinite(output_numpy).all()

    @pytest.mark.parametrize("sample_input", ["sample_input_2d", "sample_input_3d"])
    def test_forward_pass_nonsquare(self, nonsquare_config: Dict[str, Any],
                                    sample_input: str, request) -> None:
        """Test forward pass for non-square config (hidden_dim != output_dim)."""
        inputs = request.getfixturevalue(sample_input)
        layer = MonarchFFN(**nonsquare_config)

        output = layer(inputs)

        expected_shape = list(inputs.shape)
        expected_shape[-1] = nonsquare_config['output_dim']
        assert output.shape == tuple(expected_shape)

        output_numpy = keras.ops.convert_to_numpy(output)
        assert np.isfinite(output_numpy).all()

    def test_compute_output_shape(self, layer_config: Dict[str, Any]) -> None:
        """Test compute_output_shape transforms only the last dim."""
        layer = MonarchFFN(**layer_config)

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
        outputs = MonarchFFN(**layer_config)(inputs)
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
        layer = MonarchFFN(**layer_config)
        config = layer.get_config()

        required_keys = {
            'hidden_dim', 'output_dim', 'nblocks', 'activation', 'dropout_rate',
            'use_bias', 'kernel_initializer', 'bias_initializer',
            'kernel_regularizer', 'bias_regularizer'
        }
        for key in required_keys:
            assert key in config, f"Missing {key} in get_config()"

        assert config['hidden_dim'] == layer_config['hidden_dim']
        assert config['output_dim'] == layer_config['output_dim']
        assert config['nblocks'] == layer_config['nblocks']
        assert config['dropout_rate'] == layer_config['dropout_rate']
        assert config['use_bias'] == layer_config['use_bias']

    def test_gradients_flow(self, layer_config: Dict[str, Any],
                            sample_input_3d: keras.KerasTensor) -> None:
        """Test that gradients flow to all trainable weights and are non-zero."""
        layer = MonarchFFN(**layer_config)

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
        """Test error conditions and divisibility constraints."""
        # nblocks does not divide hidden_dim
        with pytest.raises(ValueError, match="hidden_dim must be divisible by nblocks"):
            MonarchFFN(hidden_dim=64, output_dim=64, nblocks=5)

        # nblocks does not divide output_dim
        with pytest.raises(ValueError, match="output_dim must be divisible by nblocks"):
            MonarchFFN(hidden_dim=64, output_dim=30, nblocks=4)

        # invalid hidden_dim
        with pytest.raises(ValueError, match="hidden_dim must be a positive integer"):
            MonarchFFN(hidden_dim=0, output_dim=64, nblocks=4)

        # invalid output_dim
        with pytest.raises(ValueError, match="output_dim must be a positive integer"):
            MonarchFFN(hidden_dim=64, output_dim=-1, nblocks=4)

        # invalid nblocks
        with pytest.raises(ValueError, match="nblocks must be a positive integer"):
            MonarchFFN(hidden_dim=64, output_dim=64, nblocks=0)

        # invalid dropout_rate
        with pytest.raises(ValueError, match="dropout_rate must be in"):
            MonarchFFN(hidden_dim=64, output_dim=64, nblocks=4, dropout_rate=1.1)

        # input_dim not divisible by nblocks raises at build
        layer = MonarchFFN(hidden_dim=64, output_dim=64, nblocks=4)
        with pytest.raises(ValueError, match="input_dim must be divisible by nblocks"):
            layer.build((None, 30))

        # undefined last dim raises at build
        layer2 = MonarchFFN(hidden_dim=64, output_dim=64, nblocks=4)
        with pytest.raises(ValueError, match="last dimension of input_shape must be defined"):
            layer2.build((None, None))

    def test_no_bias_configuration(self, sample_input_2d: keras.KerasTensor) -> None:
        """Test layer without bias terms."""
        layer = MonarchFFN(hidden_dim=64, output_dim=64, nblocks=4, use_bias=False)
        output = layer(sample_input_2d)
        assert output.shape == (4, 64)
        assert layer.expand_bias is None
        assert layer.contract_bias is None

    def test_factory_creation(self, sample_input_2d: keras.KerasTensor) -> None:
        """Test creation through the FFN factory (catches param-drop trap)."""
        layer = create_ffn_layer('monarch', hidden_dim=64, output_dim=64, nblocks=4)

        assert isinstance(layer, MonarchFFN)
        # nblocks must have survived the factory param filter
        assert layer.nblocks == 4
        assert layer.hidden_dim == 64
        assert layer.output_dim == 64

        output = layer(sample_input_2d)
        assert output.shape == (4, 64)
        output_numpy = keras.ops.convert_to_numpy(output)
        assert np.isfinite(output_numpy).all()

    def test_factory_invalid_nblocks(self) -> None:
        """Factory validation rejects non-positive nblocks."""
        with pytest.raises(ValueError):
            create_ffn_layer('monarch', hidden_dim=64, output_dim=64, nblocks=0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
