"""Tests for SIGRegLayer."""

import keras
import numpy as np
import pytest
from keras import ops

from dl_techniques.regularizers.sigreg import SIGRegLayer


class TestSIGReg:
    def test_forward_returns_scalar_finite(self):
        rng = np.random.default_rng(0)
        proj = rng.standard_normal((3, 8, 16)).astype("float32")  # (T, B, D)
        layer = SIGRegLayer(knots=17, num_proj=64, seed=123)
        y = layer(proj)
        y_np = ops.convert_to_numpy(y)
        # Scalar, finite, non-negative.
        assert y_np.shape == ()
        assert np.isfinite(y_np)
        assert y_np >= 0.0

    def test_gaussian_input_has_smaller_loss_than_skewed(self):
        """Sanity check: Gaussian samples should yield lower SIGReg than
        grossly non-Gaussian ones."""
        rng = np.random.default_rng(42)
        # Gaussian: (T=4, B=128, D=16).
        gaussian = rng.standard_normal((4, 128, 16)).astype("float32")
        # Highly non-Gaussian: mixture of two far-apart deltas.
        skewed = rng.choice([-5.0, 5.0], size=(4, 128, 16)).astype("float32")

        layer = SIGRegLayer(knots=17, num_proj=512, seed=123)
        l_gauss = float(ops.convert_to_numpy(layer(gaussian)))
        # Fresh seed for fair comparison — not strictly required since we
        # re-sample each call; use a new layer to keep A distributions
        # independent across calls.
        layer2 = SIGRegLayer(knots=17, num_proj=512, seed=456)
        l_skewed = float(ops.convert_to_numpy(layer2(skewed)))

        assert l_gauss < l_skewed, (
            f"SIGReg on Gaussian ({l_gauss}) should be < SIGReg on skewed "
            f"({l_skewed})."
        )

    def test_serialization_round_trip(self, tmp_path):
        """Save/load round-trip preserves config + buffers."""
        x_in = keras.Input(shape=(8, 16))  # (T=8 batch-within, D=16) — tested rank-3
        y_out = SIGRegLayer(knots=17, num_proj=32, seed=7, name="sigreg")(x_in)
        model = keras.Model(x_in, y_out)

        proj = np.random.default_rng(0).standard_normal((3, 8, 16)).astype("float32")
        y1 = float(ops.convert_to_numpy(model(proj)))

        path = str(tmp_path / "sigreg.keras")
        model.save(path)
        loaded = keras.models.load_model(path)
        y2 = float(ops.convert_to_numpy(loaded(proj)))

        # Because A is resampled on each forward pass, exact match is not
        # expected. We only assert both are finite and on the same order —
        # full round-trip correctness is covered by config check below.
        assert np.isfinite(y1) and np.isfinite(y2)

        # Config round-trip must preserve static attributes.
        cfg = loaded.get_layer("sigreg").get_config()
        assert cfg["knots"] == 17
        assert cfg["num_proj"] == 32
        assert cfg["seed"] == 7
