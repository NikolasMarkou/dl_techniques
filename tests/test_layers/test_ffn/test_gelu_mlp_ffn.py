import os
import tempfile

import keras
import numpy as np
import pytest
from typing import Any, Dict

from dl_techniques.layers.ffn.gelu_mlp_ffn import GELUMLPFFN


class TestGELUMLPFFN:
    """Comprehensive test suite for the GELUMLPFFN (SD3 GELU-tanh FFN) layer."""

    @pytest.fixture
    def layer_config(self) -> Dict[str, Any]:
        """Standard configuration with an explicit output_dim."""
        return {
            "hidden_dim": 256,
            "output_dim": 128,
            "dropout_rate": 0.1,
            "use_bias": True,
        }

    @pytest.fixture
    def sample_input_2d(self) -> keras.KerasTensor:
        return keras.random.normal(shape=(4, 64))

    @pytest.fixture
    def sample_input_3d(self) -> keras.KerasTensor:
        return keras.random.normal(shape=(2, 32, 64))

    # ------------------------------------------------------------------
    # (1) instantiation
    # ------------------------------------------------------------------
    def test_initialization(self, layer_config: Dict[str, Any]) -> None:
        layer = GELUMLPFFN(**layer_config)

        assert layer.hidden_dim == layer_config["hidden_dim"]
        assert layer.output_dim == layer_config["output_dim"]
        assert layer.dropout_rate == layer_config["dropout_rate"]
        assert layer.use_bias == layer_config["use_bias"]
        assert not layer.built

        assert isinstance(layer.fc1, keras.layers.Dense)
        assert isinstance(layer.fc2, keras.layers.Dense)
        assert isinstance(layer.dropout, keras.layers.Dropout)
        assert layer.fc1.units == layer_config["hidden_dim"]
        assert layer.fc2.units == layer_config["output_dim"]
        assert layer.dropout.rate == layer_config["dropout_rate"]

    def test_invalid_args(self) -> None:
        with pytest.raises(ValueError):
            GELUMLPFFN(hidden_dim=0)
        with pytest.raises(ValueError):
            GELUMLPFFN(hidden_dim=16, output_dim=-1)
        with pytest.raises(ValueError):
            GELUMLPFFN(hidden_dim=16, dropout_rate=1.5)

    # ------------------------------------------------------------------
    # (2) forward shape: explicit output_dim and output_dim=None -> input dim
    # ------------------------------------------------------------------
    @pytest.mark.parametrize("sample_input", ["sample_input_2d", "sample_input_3d"])
    def test_forward_explicit_output_dim(
        self, layer_config: Dict[str, Any], sample_input: str, request
    ) -> None:
        x = request.getfixturevalue(sample_input)
        layer = GELUMLPFFN(**layer_config)
        y = layer(x)
        assert layer.built
        assert y.shape[:-1] == x.shape[:-1]
        assert y.shape[-1] == layer_config["output_dim"]

    @pytest.mark.parametrize("sample_input", ["sample_input_2d", "sample_input_3d"])
    def test_forward_output_dim_none(self, sample_input: str, request) -> None:
        x = request.getfixturevalue(sample_input)
        layer = GELUMLPFFN(hidden_dim=128)  # output_dim None -> resolves to input dim
        y = layer(x)
        assert layer.built
        # output dim equals the input feature dim
        assert y.shape[-1] == x.shape[-1]
        assert layer.fc2.units == x.shape[-1]

    # ------------------------------------------------------------------
    # (3) GELU-tanh numerical check: proves approximate=True is used
    # ------------------------------------------------------------------
    def test_uses_approximate_gelu(self) -> None:
        """A 1-layer config must match manual tanh-approximate GELU, and must
        differ from the exact-erf GELU path on the same fixed input."""
        layer = GELUMLPFFN(hidden_dim=8, output_dim=4, dropout_rate=0.0, use_bias=True)
        x = keras.random.normal(shape=(3, 6))
        y = layer(x)  # builds the layer

        w1, b1 = layer.fc1.get_weights()
        w2, b2 = layer.fc2.get_weights()

        h = np.matmul(np.asarray(x), w1) + b1
        h_approx = np.asarray(keras.ops.gelu(keras.ops.convert_to_tensor(h), approximate=True))
        y_manual_approx = np.matmul(h_approx, w2) + b2

        h_exact = np.asarray(keras.ops.gelu(keras.ops.convert_to_tensor(h), approximate=False))
        y_manual_exact = np.matmul(h_exact, w2) + b2

        # Layer matches the tanh-approximate reference (tol loose: the manual
        # path recomputes fc1 in numpy float, drifting the cubic term slightly).
        np.testing.assert_allclose(np.asarray(y), y_manual_approx, atol=1e-3)
        # The approximate path is closer to the layer than the exact-erf path,
        # AND the two paths are measurably different -> approximate=True is used.
        err_approx = np.max(np.abs(np.asarray(y) - y_manual_approx))
        err_exact = np.max(np.abs(np.asarray(y) - y_manual_exact))
        assert err_approx < err_exact
        assert not np.allclose(y_manual_approx, y_manual_exact, atol=1e-4)

    # ------------------------------------------------------------------
    # (4) compute_output_shape pre/post build
    # ------------------------------------------------------------------
    def test_compute_output_shape_explicit(self, layer_config: Dict[str, Any]) -> None:
        layer = GELUMLPFFN(**layer_config)
        out = layer.compute_output_shape((None, 64))
        assert out == (None, layer_config["output_dim"])  # pre-build, explicit dim

        x = keras.random.normal(shape=(4, 64))
        layer(x)
        out_post = layer.compute_output_shape((None, 64))
        assert out_post == (None, layer_config["output_dim"])

    def test_compute_output_shape_none(self) -> None:
        layer = GELUMLPFFN(hidden_dim=32)
        # Pre-build with unresolved output_dim: last dim preserved.
        out_pre = layer.compute_output_shape((None, 64))
        assert out_pre == (None, 64)

        layer(keras.random.normal(shape=(2, 64)))
        out_post = layer.compute_output_shape((None, 64))
        assert out_post == (None, 64)

    # ------------------------------------------------------------------
    # (5) get_config / from_config
    # ------------------------------------------------------------------
    def test_config_round_trip(self, layer_config: Dict[str, Any]) -> None:
        layer = GELUMLPFFN(**layer_config)
        config = layer.get_config()
        for key, val in layer_config.items():
            assert config[key] == val

        rebuilt = GELUMLPFFN.from_config(config)
        assert rebuilt.hidden_dim == layer.hidden_dim
        assert rebuilt.output_dim == layer.output_dim
        assert rebuilt.dropout_rate == layer.dropout_rate
        assert rebuilt.use_bias == layer.use_bias

    def test_config_none_output_dim(self) -> None:
        layer = GELUMLPFFN(hidden_dim=64)
        config = layer.get_config()
        assert config["output_dim"] is None  # original None preserved
        rebuilt = GELUMLPFFN.from_config(config)
        assert rebuilt.output_dim is None

    # ------------------------------------------------------------------
    # (6) .keras round-trip @ atol 1e-6 (Functional model wrapper)
    # ------------------------------------------------------------------
    @pytest.mark.parametrize("output_dim", [128, None])
    def test_keras_round_trip(self, output_dim) -> None:
        inputs = keras.Input(shape=(32, 64))
        outputs = GELUMLPFFN(hidden_dim=128, output_dim=output_dim, dropout_rate=0.0)(inputs)
        model = keras.Model(inputs, outputs)

        x = keras.random.normal(shape=(2, 32, 64))
        y_ref = model(x)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "gelu_mlp_ffn.keras")
            model.save(path)
            loaded = keras.models.load_model(path)
            y_loaded = loaded(x)

        np.testing.assert_allclose(
            np.asarray(y_ref), np.asarray(y_loaded), atol=1e-6
        )

    # ------------------------------------------------------------------
    # (7) variable batch
    # ------------------------------------------------------------------
    def test_variable_batch(self, layer_config: Dict[str, Any]) -> None:
        layer = GELUMLPFFN(**layer_config)
        layer.build((None, 64))
        for batch in (1, 5, 16):
            x = keras.random.normal(shape=(batch, 64))
            y = layer(x)
            assert y.shape == (batch, layer_config["output_dim"])

    # ------------------------------------------------------------------
    # (8) factory construction
    # ------------------------------------------------------------------
    def test_factory_construction(self) -> None:
        from dl_techniques.layers.ffn.factory import create_ffn_layer, FFN_REGISTRY

        assert "gelu_tanh" in FFN_REGISTRY
        layer = create_ffn_layer("gelu_tanh", hidden_dim=64)
        assert isinstance(layer, GELUMLPFFN)
        assert layer.hidden_dim == 64
        assert layer.output_dim is None

        x = keras.random.normal(shape=(2, 16, 48))
        y = layer(x)
        assert y.shape[-1] == 48  # output_dim None -> input dim

    def test_factory_construction_explicit_output(self) -> None:
        from dl_techniques.layers.ffn.factory import create_ffn_layer

        layer = create_ffn_layer(
            "gelu_tanh", hidden_dim=64, output_dim=32, dropout_rate=0.0, use_bias=False
        )
        assert isinstance(layer, GELUMLPFFN)
        assert layer.output_dim == 32
        assert layer.use_bias is False
        y = layer(keras.random.normal(shape=(2, 64)))
        assert y.shape == (2, 32)
