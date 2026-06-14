import pytest
import tempfile
import os
import numpy as np
import keras
from typing import Any, Dict
import tensorflow as tf

from dl_techniques.layers.ffn import OrthoGLUFFN


class TestOrthoGLUFFN:
    """Comprehensive test suite for the OrthoGLUFFN layer."""

    @pytest.fixture
    def layer_config(self) -> Dict[str, Any]:
        """Standard configuration for testing."""
        return {
            'hidden_dim': 64,
            'output_dim': 32,
            'activation': 'gelu',
            'dropout_rate': 0.1,
            'use_bias': True,
            'ortho_reg_factor': 1.0,
        }

    @pytest.fixture
    def sample_input_2d(self) -> keras.KerasTensor:
        """Sample 2D input."""
        return keras.random.normal(shape=(4, 48))

    @pytest.fixture
    def sample_input_3d(self) -> keras.KerasTensor:
        """Sample 3D input (sequence data)."""
        return keras.random.normal(shape=(2, 16, 48))

    @pytest.fixture
    def sample_input_4d(self) -> keras.KerasTensor:
        """Sample 4D input (image-like data)."""
        return keras.random.normal(shape=(2, 8, 8, 48))

    def test_initialization(self, layer_config: Dict[str, Any]) -> None:
        """Test layer initialization and sub-layer creation."""
        layer = OrthoGLUFFN(**layer_config)

        assert layer.hidden_dim == layer_config['hidden_dim']
        assert layer.output_dim == layer_config['output_dim']
        assert layer.dropout_rate == layer_config['dropout_rate']
        assert layer.use_bias == layer_config['use_bias']
        assert layer.ortho_reg_factor == layer_config['ortho_reg_factor']

        # Not built yet
        assert not layer.built

        # Sub-layers created in __init__
        assert hasattr(layer, 'input_proj_ortho')
        assert hasattr(layer, 'output_proj_ortho')
        assert hasattr(layer, 'dropout')
        assert isinstance(layer.dropout, keras.layers.Dropout)

        # input ortho projects to 2 * hidden_dim (gate + value)
        assert layer.input_proj_ortho.units == layer_config['hidden_dim'] * 2
        assert layer.output_proj_ortho.units == layer_config['output_dim']

    @pytest.mark.parametrize("sample_input", [
        "sample_input_2d", "sample_input_3d", "sample_input_4d"
    ])
    def test_forward_pass(self, layer_config: Dict[str, Any],
                          sample_input: str, request) -> None:
        """Forward pass + building across ranks."""
        inputs = request.getfixturevalue(sample_input)
        layer = OrthoGLUFFN(**layer_config)

        output = layer(inputs)

        assert layer.built
        assert layer.input_proj_ortho.built
        assert layer.output_proj_ortho.built

        expected_shape = list(inputs.shape)
        expected_shape[-1] = layer_config['output_dim']
        assert output.shape == tuple(expected_shape)

        output_numpy = keras.ops.convert_to_numpy(output)
        assert not np.isnan(output_numpy).any()
        assert not np.isinf(output_numpy).any()

    def test_serialization_cycle(self, layer_config: Dict[str, Any],
                                 sample_input_3d: keras.KerasTensor) -> None:
        """CRITICAL: full .keras round-trip inside a keras.Model."""
        inputs = keras.Input(shape=sample_input_3d.shape[1:])
        outputs = OrthoGLUFFN(**layer_config)(inputs)
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

    def test_tuple_ortho_reg_factor_roundtrip(
            self, sample_input_2d: keras.KerasTensor) -> None:
        """A tuple ortho_reg_factor must survive a .keras round-trip.

        Keras serializes tuples to JSON lists; the layer must still rebuild and
        produce identical outputs.
        """
        config = dict(hidden_dim=64, output_dim=32, ortho_reg_factor=(0.5, 2.0))
        inputs = keras.Input(shape=sample_input_2d.shape[1:])
        outputs = OrthoGLUFFN(**config)(inputs)
        model = keras.Model(inputs, outputs)
        original = model(sample_input_2d)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'tuple_model.keras')
            model.save(filepath)
            loaded = keras.models.load_model(filepath)
            reloaded = loaded(sample_input_2d)

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(original),
            keras.ops.convert_to_numpy(reloaded),
            rtol=1e-6, atol=1e-6,
            err_msg="Tuple ortho_reg_factor broke the round-trip"
        )

    def test_config_completeness(self, layer_config: Dict[str, Any]) -> None:
        """get_config must contain all __init__ params and rebuild."""
        layer = OrthoGLUFFN(**layer_config)
        config = layer.get_config()

        required_keys = {
            'hidden_dim', 'output_dim', 'activation', 'dropout_rate',
            'use_bias', 'ortho_reg_factor',
        }
        for key in required_keys:
            assert key in config, f"Missing {key} in get_config()"

        assert config['hidden_dim'] == layer_config['hidden_dim']
        assert config['output_dim'] == layer_config['output_dim']
        assert config['dropout_rate'] == layer_config['dropout_rate']
        assert config['use_bias'] == layer_config['use_bias']

        # from_config reconstruction
        rebuilt = OrthoGLUFFN.from_config(config)
        assert rebuilt.hidden_dim == layer_config['hidden_dim']
        assert rebuilt.output_dim == layer_config['output_dim']

    def test_gradients_flow(self, layer_config: Dict[str, Any],
                            sample_input_3d: keras.KerasTensor) -> None:
        """Gradients flow to inputs and weights."""
        layer = OrthoGLUFFN(**layer_config)

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

    def test_ortho_regularization_losses(
            self, layer_config: Dict[str, Any],
            sample_input_2d: keras.KerasTensor) -> None:
        """OrthoBlock regularization should contribute to layer.losses."""
        layer = OrthoGLUFFN(**layer_config)
        _ = layer(sample_input_2d)
        assert len(layer.losses) > 0

    @pytest.mark.parametrize("training_mode", [True, False, None])
    def test_training_modes(self, layer_config: Dict[str, Any],
                            sample_input_3d: keras.KerasTensor,
                            training_mode: bool) -> None:
        """Output shape consistent across training modes; eval is deterministic."""
        layer = OrthoGLUFFN(**layer_config)
        output = layer(sample_input_3d, training=training_mode)

        expected_shape = list(sample_input_3d.shape)
        expected_shape[-1] = layer_config['output_dim']
        assert output.shape == tuple(expected_shape)

        if training_mode is False:
            output2 = layer(sample_input_3d, training=False)
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(output),
                keras.ops.convert_to_numpy(output2),
                rtol=1e-6, atol=1e-6,
                err_msg="Evaluation mode outputs should be deterministic"
            )

    def test_edge_cases(self) -> None:
        """Error conditions."""
        with pytest.raises(ValueError, match="hidden_dim must be positive"):
            OrthoGLUFFN(hidden_dim=0, output_dim=32)

        with pytest.raises(ValueError, match="output_dim must be positive"):
            OrthoGLUFFN(hidden_dim=32, output_dim=-1)

        with pytest.raises(ValueError, match="dropout_rate must be between 0 and 1"):
            OrthoGLUFFN(hidden_dim=32, output_dim=16, dropout_rate=1.5)

        layer = OrthoGLUFFN(hidden_dim=32, output_dim=16)
        with pytest.raises(ValueError, match="Last dimension of input must be defined"):
            layer.build((None, None))

    def test_compute_output_shape(self, layer_config: Dict[str, Any]) -> None:
        """compute_output_shape sets last dim to output_dim, preserves the rest."""
        layer = OrthoGLUFFN(**layer_config)
        test_shapes = [
            (None, 48),
            (None, 16, 48),
            (None, 8, 8, 48),
        ]
        for input_shape in test_shapes:
            output_shape = layer.compute_output_shape(input_shape)
            assert output_shape[-1] == layer_config['output_dim']
            assert output_shape[:-1] == input_shape[:-1]

    def test_build_idempotent(self, layer_config: Dict[str, Any],
                              sample_input_2d: keras.KerasTensor) -> None:
        """A second build() must not rebuild sub-layers or change outputs."""
        layer = OrthoGLUFFN(**layer_config)
        out1 = layer(sample_input_2d, training=False)
        # Re-invoke build explicitly; the guard must make this a no-op.
        layer.build(sample_input_2d.shape)
        out2 = layer(sample_input_2d, training=False)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(out1),
            keras.ops.convert_to_numpy(out2),
            rtol=1e-6, atol=1e-6,
            err_msg="Second build() changed the output (guard missing/broken)"
        )

    def test_gating_mechanism(self, sample_input_2d: keras.KerasTensor) -> None:
        """The layer output must equal a manual reconstruction of the GLU path."""
        layer = OrthoGLUFFN(
            hidden_dim=32,
            output_dim=16,
            activation='sigmoid',
            dropout_rate=0.0,
        )
        output = layer(sample_input_2d, training=False)

        gate_and_value = layer.input_proj_ortho(sample_input_2d, training=False)
        gate, value = keras.ops.split(gate_and_value, indices_or_sections=2, axis=-1)
        manual = layer.output_proj_ortho(
            keras.activations.sigmoid(gate) * value, training=False
        )

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(output),
            keras.ops.convert_to_numpy(manual),
            rtol=1e-5, atol=1e-5,
            err_msg="Gating mechanism reconstruction mismatch"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
