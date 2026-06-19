import pytest
import tempfile
import os
import numpy as np
import keras
from typing import Any, Dict
import tensorflow as tf

from dl_techniques.layers.ffn.mlp_mixer_block import MixerBlock
from dl_techniques.layers.ffn.factory import create_ffn_layer


class TestMixerBlock:
    """Comprehensive test suite for the MixerBlock layer (rank-3 only)."""

    @pytest.fixture
    def layer_config(self) -> Dict[str, Any]:
        """Standard configuration for testing."""
        return {
            'tokens_mlp_dim': 32,
            'channels_mlp_dim': 128,
            'activation': 'gelu',
            'dropout_rate': 0.1,
            'use_bias': True,
            'kernel_initializer': 'glorot_uniform',
            'bias_initializer': 'zeros',
        }

    @pytest.fixture
    def sample_input_3d(self) -> keras.KerasTensor:
        """Sample rank-3 input (B, S, C)."""
        return keras.random.normal(shape=(2, 16, 64))

    def test_initialization(self, layer_config: Dict[str, Any]) -> None:
        """Test layer initialization stores all params (catches factory param-drop)."""
        layer = MixerBlock(**layer_config)

        assert layer.tokens_mlp_dim == layer_config['tokens_mlp_dim']
        assert layer.channels_mlp_dim == layer_config['channels_mlp_dim']
        assert layer.dropout_rate == layer_config['dropout_rate']
        assert layer.use_bias == layer_config['use_bias']
        assert layer.activation is not None

        # Not built yet; dimension-dependent projections created in build().
        assert not layer.built
        assert layer.token_mlp_out is None
        assert layer.channel_mlp_out is None

        # Config-known sublayers created in __init__.
        assert isinstance(layer.token_norm, keras.layers.LayerNormalization)
        assert isinstance(layer.channel_norm, keras.layers.LayerNormalization)
        assert isinstance(layer.token_mlp_hidden, keras.layers.Dense)
        assert isinstance(layer.channel_mlp_hidden, keras.layers.Dense)
        assert isinstance(layer.token_dropout, keras.layers.Dropout)
        assert isinstance(layer.channel_dropout, keras.layers.Dropout)
        assert layer.token_dropout.rate == layer_config['dropout_rate']

    def test_forward_pass(self, layer_config: Dict[str, Any],
                          sample_input_3d: keras.KerasTensor) -> None:
        """Test forward pass: output shape == input shape, finite values."""
        layer = MixerBlock(**layer_config)

        output = layer(sample_input_3d)

        assert layer.built
        # Output shape identical to input shape.
        assert output.shape == sample_input_3d.shape == (2, 16, 64)

        # Dimension-dependent projections were created.
        assert layer.token_mlp_out is not None
        assert layer.channel_mlp_out is not None

        output_numpy = keras.ops.convert_to_numpy(output)
        assert np.isfinite(output_numpy).all()

    def test_rank_contract(self, layer_config: Dict[str, Any]) -> None:
        """Test that non-rank-3 inputs raise ValueError at build."""
        # rank-2 input
        layer_2d = MixerBlock(**layer_config)
        with pytest.raises(ValueError, match="rank-3"):
            layer_2d(keras.random.normal(shape=(4, 64)))

        # rank-4 input
        layer_4d = MixerBlock(**layer_config)
        with pytest.raises(ValueError, match="rank-3"):
            layer_4d(keras.random.normal(shape=(2, 8, 8, 64)))

    def test_compute_output_shape(self, layer_config: Dict[str, Any]) -> None:
        """Test compute_output_shape returns input shape unchanged."""
        layer = MixerBlock(**layer_config)

        test_shapes = [
            (None, 16, 64),
            (2, 16, 64),
            (8, 49, 256),
        ]
        for input_shape in test_shapes:
            output_shape = layer.compute_output_shape(input_shape)
            assert output_shape == tuple(input_shape)

    def test_serialization_cycle(self, layer_config: Dict[str, Any],
                                 sample_input_3d: keras.KerasTensor) -> None:
        """CRITICAL: full .keras serialization cycle with prediction comparison."""
        inputs = keras.Input(shape=sample_input_3d.shape[1:])
        outputs = MixerBlock(**layer_config)(inputs)
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
        layer = MixerBlock(**layer_config)
        config = layer.get_config()

        required_keys = {
            'tokens_mlp_dim', 'channels_mlp_dim', 'activation', 'dropout_rate',
            'use_bias', 'kernel_initializer', 'bias_initializer',
            'kernel_regularizer', 'bias_regularizer'
        }
        for key in required_keys:
            assert key in config, f"Missing {key} in get_config()"

        assert config['tokens_mlp_dim'] == layer_config['tokens_mlp_dim']
        assert config['channels_mlp_dim'] == layer_config['channels_mlp_dim']
        assert config['dropout_rate'] == layer_config['dropout_rate']
        assert config['use_bias'] == layer_config['use_bias']

    def test_gradients_flow(self, layer_config: Dict[str, Any],
                            sample_input_3d: keras.KerasTensor) -> None:
        """Test that gradients flow to all trainable weights and are non-zero."""
        layer = MixerBlock(**layer_config)

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
        """Test error conditions in the constructor."""
        with pytest.raises(ValueError, match="tokens_mlp_dim must be a positive integer"):
            MixerBlock(tokens_mlp_dim=0, channels_mlp_dim=128)

        with pytest.raises(ValueError, match="channels_mlp_dim must be a positive integer"):
            MixerBlock(tokens_mlp_dim=32, channels_mlp_dim=-1)

        with pytest.raises(ValueError, match="dropout_rate must be"):
            MixerBlock(tokens_mlp_dim=32, channels_mlp_dim=128, dropout_rate=1.1)

    def test_factory_creation(self, sample_input_3d: keras.KerasTensor) -> None:
        """Test creation through the FFN factory (catches param-drop trap)."""
        layer = create_ffn_layer('mixer', tokens_mlp_dim=32, channels_mlp_dim=128)

        assert isinstance(layer, MixerBlock)
        assert layer.tokens_mlp_dim == 32
        assert layer.channels_mlp_dim == 128

        output = layer(sample_input_3d)
        assert output.shape == (2, 16, 64)
        output_numpy = keras.ops.convert_to_numpy(output)
        assert np.isfinite(output_numpy).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
