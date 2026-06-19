import pytest
import tempfile
import os
import numpy as np
import keras
from typing import Any, Dict
import tensorflow as tf

from dl_techniques.layers.ffn.lowrank_ffn import LowRankFFN
from dl_techniques.layers.ffn.mlp import MLPBlock
from dl_techniques.layers.ffn.factory import create_ffn_layer


class TestLowRankFFN:
    """Comprehensive test suite for the LowRankFFN layer."""

    @pytest.fixture
    def layer_config(self) -> Dict[str, Any]:
        """Standard configuration for testing."""
        return {
            'hidden_dim': 128,
            'output_dim': 64,
            'rank': 16,
            'activation': 'gelu',
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
        layer = LowRankFFN(**layer_config)

        assert layer.hidden_dim == layer_config['hidden_dim']
        assert layer.output_dim == layer_config['output_dim']
        assert layer.rank == layer_config['rank']
        assert layer._rank_arg == layer_config['rank']
        assert layer.dropout_rate == layer_config['dropout_rate']
        assert layer.use_bias == layer_config['use_bias']

        # Not built yet; Dense weights created on build().
        assert not layer.built

        # Sub-layers created in __init__.
        assert isinstance(layer.u1, keras.layers.Dense)
        assert isinstance(layer.v1, keras.layers.Dense)
        assert isinstance(layer.u2, keras.layers.Dense)
        assert isinstance(layer.v2, keras.layers.Dense)
        assert isinstance(layer.dropout, keras.layers.Dropout)

        # Bottleneck (U) widths == rank, bias-free; V projections carry dims + bias.
        assert layer.u1.units == layer_config['rank']
        assert layer.v1.units == layer_config['hidden_dim']
        assert layer.u2.units == layer_config['rank']
        assert layer.v2.units == layer_config['output_dim']
        assert layer.u1.use_bias is False
        assert layer.u2.use_bias is False
        assert layer.v1.use_bias == layer_config['use_bias']
        assert layer.v2.use_bias == layer_config['use_bias']

    def test_rank_default_resolution(self) -> None:
        """rank=None resolves to max(1, hidden_dim // 4) at construction time."""
        layer = LowRankFFN(hidden_dim=128, output_dim=64, rank=None)
        assert layer._rank_arg is None
        assert layer.rank == max(1, 128 // 4)
        assert layer.rank == 32
        assert layer.u1.units == 32
        assert layer.u2.units == 32

        # Tiny hidden_dim floors at 1.
        layer_small = LowRankFFN(hidden_dim=2, output_dim=4)
        assert layer_small.rank == max(1, 2 // 4)
        assert layer_small.rank == 1

    @pytest.mark.parametrize("sample_input,leading", [
        ("sample_input_2d", (4,)),
        ("sample_input_3d", (2, 16)),
    ])
    def test_forward_pass(self, layer_config: Dict[str, Any],
                          sample_input: str, leading, request) -> None:
        """Test forward pass + build on 2D / 3D inputs."""
        inputs = request.getfixturevalue(sample_input)
        layer = LowRankFFN(**layer_config)

        output = layer(inputs)

        assert layer.built

        expected_shape = (*leading, layer_config['output_dim'])
        assert output.shape == expected_shape
        assert output.shape[-1] == layer_config['output_dim']

        output_numpy = keras.ops.convert_to_numpy(output)
        assert np.isfinite(output_numpy).all()

    def test_param_count(self) -> None:
        """Low-rank FFN has fewer params than an equivalent dense MLP."""
        hidden_dim, output_dim, rank = 256, 256, 8
        input_dim = 256

        lowrank = LowRankFFN(hidden_dim=hidden_dim, output_dim=output_dim, rank=rank)
        dense = MLPBlock(hidden_dim=hidden_dim, output_dim=output_dim)

        # Build both on the same input shape.
        dummy = keras.random.normal(shape=(2, input_dim))
        _ = lowrank(dummy)
        _ = dense(dummy)

        lowrank_params = lowrank.count_params()
        dense_params = dense.count_params()

        assert lowrank_params < dense_params, (
            f"Expected low-rank ({lowrank_params}) < dense ({dense_params})"
        )

    def test_compute_output_shape(self, layer_config: Dict[str, Any]) -> None:
        """Test compute_output_shape transforms only the last dim."""
        layer = LowRankFFN(**layer_config)

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
        outputs = LowRankFFN(**layer_config)(inputs)
        model = keras.Model(inputs, outputs)

        original_prediction = model(sample_input_3d)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_model.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_prediction = loaded_model(sample_input_3d)

            # rank must round-trip: locate the reloaded layer and check it.
            reloaded_layer = next(
                lyr for lyr in loaded_model.layers if isinstance(lyr, LowRankFFN)
            )
            assert reloaded_layer.rank == layer_config['rank']
            assert reloaded_layer._rank_arg == layer_config['rank']

            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_prediction),
                keras.ops.convert_to_numpy(loaded_prediction),
                rtol=1e-6, atol=1e-6,
                err_msg="Predictions differ after serialization"
            )

    def test_serialization_cycle_rank_none(self,
                                            sample_input_3d: keras.KerasTensor) -> None:
        """rank=None round-trips: resolved rank reconstructs identically."""
        inputs = keras.Input(shape=sample_input_3d.shape[1:])
        layer = LowRankFFN(hidden_dim=128, output_dim=64, rank=None)
        outputs = layer(inputs)
        model = keras.Model(inputs, outputs)

        original_prediction = model(sample_input_3d)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_model_none.keras')
            model.save(filepath)
            loaded_model = keras.models.load_model(filepath)
            loaded_prediction = loaded_model(sample_input_3d)

            reloaded_layer = next(
                lyr for lyr in loaded_model.layers if isinstance(lyr, LowRankFFN)
            )
            assert reloaded_layer._rank_arg is None
            assert reloaded_layer.rank == max(1, 128 // 4)

            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_prediction),
                keras.ops.convert_to_numpy(loaded_prediction),
                rtol=1e-6, atol=1e-6,
                err_msg="Predictions differ after serialization (rank=None)"
            )

    def test_config_completeness(self, layer_config: Dict[str, Any]) -> None:
        """Test that get_config contains every __init__ param."""
        layer = LowRankFFN(**layer_config)
        config = layer.get_config()

        required_keys = {
            'hidden_dim', 'output_dim', 'rank', 'activation', 'dropout_rate',
            'use_bias', 'kernel_initializer', 'bias_initializer',
            'kernel_regularizer', 'bias_regularizer'
        }
        for key in required_keys:
            assert key in config, f"Missing {key} in get_config()"

        assert config['hidden_dim'] == layer_config['hidden_dim']
        assert config['output_dim'] == layer_config['output_dim']
        assert config['rank'] == layer_config['rank']
        assert config['dropout_rate'] == layer_config['dropout_rate']
        assert config['use_bias'] == layer_config['use_bias']

        # rank=None must be preserved as None (the as-passed arg).
        layer_none = LowRankFFN(hidden_dim=128, output_dim=64, rank=None)
        assert layer_none.get_config()['rank'] is None

    def test_gradients_flow(self, layer_config: Dict[str, Any],
                            sample_input_3d: keras.KerasTensor) -> None:
        """Test that gradients flow to all trainable weights and are non-zero."""
        layer = LowRankFFN(**layer_config)

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
            LowRankFFN(hidden_dim=0, output_dim=64)

        with pytest.raises(ValueError, match="output_dim must be positive"):
            LowRankFFN(hidden_dim=64, output_dim=-1)

        with pytest.raises(ValueError, match="rank must be positive"):
            LowRankFFN(hidden_dim=64, output_dim=64, rank=0)

        with pytest.raises(ValueError, match="dropout_rate must be in"):
            LowRankFFN(hidden_dim=64, output_dim=64, dropout_rate=1.1)

        # rank=0 must also be rejected through the factory validator.
        with pytest.raises(ValueError, match="rank must be positive"):
            create_ffn_layer('lowrank', hidden_dim=128, output_dim=64, rank=0)

    def test_factory_creation(self, sample_input_2d: keras.KerasTensor) -> None:
        """Test creation through the FFN factory (catches param-drop trap)."""
        layer = create_ffn_layer('lowrank', hidden_dim=128, output_dim=64, rank=16)

        assert isinstance(layer, LowRankFFN)
        assert layer.hidden_dim == 128
        assert layer.output_dim == 64
        assert layer.rank == 16
        assert layer._rank_arg == 16

        output = layer(sample_input_2d)
        assert output.shape == (4, 64)
        output_numpy = keras.ops.convert_to_numpy(output)
        assert np.isfinite(output_numpy).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
