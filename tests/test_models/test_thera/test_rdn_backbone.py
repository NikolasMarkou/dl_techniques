"""Scoped tests for THERA's RDN feature backbone (``rdn_backbone.py``)."""

import os
import tempfile

import keras
import numpy as np
import pytest

from dl_techniques.models.thera.rdn_backbone import RDNBackbone, RDB, RDBConv


class TestRDNBackbone:
    """Forward, shape, serialization and validation tests for RDNBackbone."""

    def test_forward_config_b(self) -> None:
        """(2, 24, 24, 3) -> (2, 24, 24, 64) with finite values (default 'B')."""
        layer = RDNBackbone()  # config 'B', G0=64
        x = keras.random.normal((2, 24, 24, 3))
        y = layer(x)
        assert tuple(y.shape) == (2, 24, 24, 64)
        assert np.all(np.isfinite(keras.ops.convert_to_numpy(y)))

    def test_forward_config_a(self) -> None:
        """config 'A' also yields G0 output channels and finite values."""
        layer = RDNBackbone(config="A")
        x = keras.random.normal((2, 16, 16, 3))
        y = layer(x)
        assert tuple(y.shape) == (2, 16, 16, 64)
        assert np.all(np.isfinite(keras.ops.convert_to_numpy(y)))

    def test_non_square(self) -> None:
        """Non-square input (1, 16, 24, 3) -> (1, 16, 24, 64)."""
        layer = RDNBackbone(config="A")
        x = keras.random.normal((1, 16, 24, 3))
        y = layer(x)
        assert tuple(y.shape) == (1, 16, 24, 64)

    def test_keras_roundtrip(self) -> None:
        """.keras save/reload preserves the forward pass to within 1e-5."""
        inp = keras.Input(shape=(16, 16, 3))
        out = RDNBackbone(config="A")(inp)
        model = keras.Model(inp, out)

        x = keras.random.normal((2, 16, 16, 3))
        y_before = keras.ops.convert_to_numpy(model(x))

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "rdn.keras")
            model.save(path)
            reloaded = keras.models.load_model(path)
            y_after = keras.ops.convert_to_numpy(reloaded(x))

        np.testing.assert_allclose(y_before, y_after, atol=1e-5)
        # Weights survive the round-trip.
        assert len(reloaded.weights) == len(model.weights)
        for w_b, w_a in zip(model.weights, reloaded.weights):
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(w_b),
                keras.ops.convert_to_numpy(w_a),
                atol=1e-6,
            )

    def test_get_config_roundtrip(self) -> None:
        """get_config/from_config reproduces the ctor args."""
        layer = RDNBackbone(growth_rate_0=48, kernel_size=3, config="A")
        cfg = layer.get_config()
        assert cfg["growth_rate_0"] == 48
        assert cfg["kernel_size"] == 3
        assert cfg["config"] == "A"

        rebuilt = RDNBackbone.from_config(cfg)
        assert rebuilt.growth_rate_0 == 48
        assert rebuilt.kernel_size == 3
        assert rebuilt.config == "A"
        # Derived (D, C, G) preset is consistent.
        assert rebuilt.num_rdb == layer.num_rdb
        assert rebuilt.num_conv_layers == layer.num_conv_layers
        assert rebuilt.growth_rate == layer.growth_rate

    def test_config_validation(self) -> None:
        """An unknown config raises ValueError."""
        with pytest.raises(ValueError):
            RDNBackbone(config="Z")

    def test_compute_output_shape(self) -> None:
        """compute_output_shape returns (B, H, W, G0)."""
        layer = RDNBackbone(growth_rate_0=64, config="A")
        assert layer.compute_output_shape((None, 32, 40, 3)) == (None, 32, 40, 64)


class TestRDBConv:
    """Channel-growth sanity for the dense conv unit."""

    def test_channel_growth(self) -> None:
        """RDBConv with growth_rate G on C-channel input outputs C + G channels."""
        c_in, g = 17, 11
        unit = RDBConv(growth_rate=g)
        x = keras.random.normal((1, 8, 8, c_in))
        y = unit(x)
        assert tuple(y.shape)[-1] == c_in + g
        assert unit.compute_output_shape((None, 8, 8, c_in))[-1] == c_in + g


class TestRDB:
    """Residual-consistent shape for a single residual dense block."""

    def test_residual_shape(self) -> None:
        """RDB maps G0-channel input back to G0-channel output."""
        block = RDB(growth_rate_0=64, growth_rate=32, num_conv_layers=4)
        x = keras.random.normal((1, 12, 12, 64))
        y = block(x)
        assert tuple(y.shape) == (1, 12, 12, 64)
