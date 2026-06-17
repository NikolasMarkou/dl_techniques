"""
Test suite for PowerMLP.

PowerMLP(hidden_units=[in, h1, ..., out]) builds a ReLU-k power MLP. Covers
construction (incl. ValueError paths), the from_variant classmethod, a forward
pass, and the M2 full .keras save -> load -> identical-output round-trip.
"""

import os
import keras
import pytest
import numpy as np

from dl_techniques.models.power_mlp.model import PowerMLP

IN_DIM = 10


def _model():
    return PowerMLP(hidden_units=[16, 8, 4])


def _x(batch=2):
    return np.random.default_rng(0).random((batch, IN_DIM)).astype("float32")


class TestPowerMLP:

    def test_forward_shape(self):
        out = _model()(_x(), training=False)
        y = out[0] if isinstance(out, (list, tuple)) else out
        assert tuple(y.shape) == (2, 4)

    def test_from_variant(self):
        model = PowerMLP.from_variant("small", num_classes=5, input_dim=IN_DIM)
        out = model(_x(), training=False)
        y = out[0] if isinstance(out, (list, tuple)) else out
        assert int(y.shape[-1]) == 5

    def test_too_few_units_raises(self):
        with pytest.raises(ValueError, match="hidden_units"):
            PowerMLP(hidden_units=[8])

    def test_nonpositive_units_raises(self):
        with pytest.raises(ValueError):
            PowerMLP(hidden_units=[16, 0, 4])

    def test_invalid_dropout_raises(self):
        with pytest.raises(ValueError, match="dropout_rate"):
            PowerMLP(hidden_units=[16, 8, 4], dropout_rate=1.5)

    def test_keras_round_trip(self, tmp_path):
        model = _model()
        x = _x()
        out_b = model(x, training=False)
        before = keras.ops.convert_to_numpy(
            out_b[0] if isinstance(out_b, (list, tuple)) else out_b)

        path = os.path.join(str(tmp_path), "power_mlp.keras")
        model.save(path)
        loaded = keras.models.load_model(path)
        out_a = loaded(x, training=False)
        after = keras.ops.convert_to_numpy(
            out_a[0] if isinstance(out_a, (list, tuple)) else out_a)

        # GPU fp32 reduction noise -> atol 1e-4 (SYSTEM invariant)
        np.testing.assert_allclose(before, after, atol=1e-4,
                                   err_msg="PowerMLP differs after .keras round-trip")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
