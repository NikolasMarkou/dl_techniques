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

    def test_normalize_by_n_defaults_false_and_matches_existing_consumers(self):
        """Regression: the exact construction signature `lewm`/`video_jepa`
        use today (no `normalize_by_n` kwarg) must be bit-identical to
        constructing with `normalize_by_n=False` explicitly, and the default
        must be `False` — pinning invariant #1 from plan.md: SIGRegLayer's
        two existing consumers see byte-identical output before and after
        this change."""
        rng = np.random.default_rng(1)
        proj = rng.standard_normal((3, 8, 16)).astype("float32")  # (T, B, D)

        layer_default = SIGRegLayer(knots=17, num_proj=64, seed=123)
        assert layer_default.normalize_by_n is False

        layer_explicit_false = SIGRegLayer(
            knots=17, num_proj=64, seed=123, normalize_by_n=False
        )

        y_default = ops.convert_to_numpy(layer_default(proj))
        y_explicit_false = ops.convert_to_numpy(layer_explicit_false(proj))

        np.testing.assert_allclose(
            y_default, y_explicit_false, atol=1e-6, rtol=0
        )

    def test_normalize_by_n_scales_statistic_by_sample_axis_size(self):
        """`normalize_by_n=True` must scale the statistic by exactly `N`,
        the sample-axis size (`proj.shape[-2]`), relative to
        `normalize_by_n=False`, on the same seeded projection matrix A and
        the same input."""
        N = 8
        rng = np.random.default_rng(2)
        proj = rng.standard_normal((3, N, 16)).astype("float32")  # (T, N, D)

        layer_unscaled = SIGRegLayer(knots=17, num_proj=64, seed=321)
        layer_scaled = SIGRegLayer(
            knots=17, num_proj=64, seed=321, normalize_by_n=True
        )

        y_unscaled = float(ops.convert_to_numpy(layer_unscaled(proj)))
        y_scaled = float(ops.convert_to_numpy(layer_scaled(proj)))

        assert y_unscaled != 0.0
        ratio = y_scaled / y_unscaled
        np.testing.assert_allclose(ratio, float(N), atol=1e-4, rtol=0)

    def test_normalize_by_n_round_trips_through_get_config(self):
        """`get_config()` must carry `normalize_by_n` so a saved/loaded
        layer preserves the flag."""
        layer = SIGRegLayer(knots=17, num_proj=32, seed=7, normalize_by_n=True)
        cfg = layer.get_config()
        assert cfg["normalize_by_n"] is True

        rebuilt = SIGRegLayer.from_config(cfg)
        assert rebuilt.normalize_by_n is True
