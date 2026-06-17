"""
Test suite for the explicit-Laplacian-pyramid CliffordNet autoencoder
(autoencoder.py).

This file currently covers the standalone ``LaplacianPyramidLevel`` helper: the
falsifiable reconstruction identity ``merge(split(x)) == x`` (atol 1e-5), the
split band shapes, and ``compute_output_shape``. Model-level tests (config
round-trip, forward, .keras save/load, variants, fit-smoke) are added in later
steps.
"""

import os

import keras
import pytest
import numpy as np

from dl_techniques.models.cliffordnet.autoencoder import (
    LaplacianPyramidLevel,
    CliffordLaplacianUNet,
    create_clifford_laplacian_unet,
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _tiny_model(**overrides) -> CliffordLaplacianUNet:
    """Smallest CPU-friendly model: 2 levels => inputs divisible by 4."""
    cfg = dict(in_channels=3, level_channels=(8, 16), level_blocks=(1, 1))
    cfg.update(overrides)
    return CliffordLaplacianUNet(**cfg)


# ---------------------------------------------------------------------
# LaplacianPyramidLevel
# ---------------------------------------------------------------------

class TestLaplacianIdentity:

    def test_reconstruction_identity(self):
        x = np.random.RandomState(0).randn(2, 64, 64, 3).astype("float32")
        layer = LaplacianPyramidLevel()
        low, high = layer(x)
        recon = layer.merge(low, high)
        recon_np = keras.ops.convert_to_numpy(recon)
        np.testing.assert_allclose(
            recon_np, x, atol=1e-5,
            err_msg="LaplacianPyramidLevel merge(split(x)) != x",
        )

    def test_split_shapes(self):
        x = np.random.RandomState(0).randn(2, 64, 64, 3).astype("float32")
        layer = LaplacianPyramidLevel()
        low, high = layer(x)
        assert tuple(low.shape) == (2, 32, 32, 3)
        assert tuple(high.shape) == (2, 64, 64, 3)

    def test_compute_output_shape(self):
        layer = LaplacianPyramidLevel()
        low_s, high_s = layer.compute_output_shape((2, 64, 64, 3))
        assert low_s == (2, 32, 32, 3)
        assert high_s == (2, 64, 64, 3)


# ---------------------------------------------------------------------
# Config round-trip
# ---------------------------------------------------------------------

class TestConfig:

    def test_config_roundtrip(self):
        m = _tiny_model()
        cfg = m.get_config()
        m2 = CliffordLaplacianUNet.from_config(cfg)
        assert m2.in_channels == m.in_channels
        # tuples -> lists in config; compare as lists.
        assert list(m2.level_channels) == list(m.level_channels)
        assert list(m2.level_blocks) == list(m.level_blocks)
        assert m2.cli_mode == m.cli_mode
        assert m2.ctx_mode == m.ctx_mode


# ---------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------

class TestForward:

    def _model3(self):
        return _tiny_model(level_channels=(8, 16, 32), level_blocks=(1, 1, 1))

    def test_forward_shape_64(self):
        m = self._model3()
        x = np.random.RandomState(0).randn(2, 64, 64, 3).astype("float32")
        out = m(x)
        assert tuple(out["reconstruction"].shape) == (2, 64, 64, 3)

    def test_forward_shape_32(self):
        m = self._model3()
        x = np.random.RandomState(1).randn(2, 32, 32, 3).astype("float32")
        out = m(x)
        assert tuple(out["reconstruction"].shape) == (2, 32, 32, 3)

    def test_compute_output_shape(self):
        m = self._model3()
        assert m.compute_output_shape((2, 64, 64, 3)) == {
            "reconstruction": (2, 64, 64, 3)
        }

    def test_divisibility_guard(self):
        m = self._model3()  # 3 levels => requires divisibility by 8
        with pytest.raises(ValueError):
            m(np.zeros((1, 30, 30, 3), dtype="float32"))


# ---------------------------------------------------------------------
# .keras save/load round-trip (PM2)
# ---------------------------------------------------------------------

class TestSaveLoad:

    def test_keras_roundtrip(self, tmp_path):
        m = _tiny_model(level_channels=(8, 16, 32), level_blocks=(1, 1, 1))
        x = np.random.RandomState(2).randn(2, 64, 64, 3).astype("float32")
        y1 = keras.ops.convert_to_numpy(m(x)["reconstruction"])

        path = os.path.join(str(tmp_path), "m.keras")
        m.save(path)
        m2 = keras.models.load_model(path)
        y2 = keras.ops.convert_to_numpy(m2(x)["reconstruction"])

        max_abs_diff = float(np.max(np.abs(y1 - y2)))
        print(f"\n[TestSaveLoad] .keras round-trip max abs diff = {max_abs_diff:.3e}")
        np.testing.assert_allclose(
            y1, y2, atol=1e-4,
            err_msg="CliffordLaplacianUNet differs after .keras round-trip",
        )


# ---------------------------------------------------------------------
# Variants + factory + imports
# ---------------------------------------------------------------------

class TestVariants:

    def test_from_variant_small(self):
        m = create_clifford_laplacian_unet("small")  # 3 levels; 64/8 ok
        x = np.random.RandomState(3).randn(1, 64, 64, 3).astype("float32")
        out = m(x)
        assert tuple(out["reconstruction"].shape) == (1, 64, 64, 3)

    def test_bad_variant(self):
        with pytest.raises(ValueError):
            CliffordLaplacianUNet.from_variant("nope")


# ---------------------------------------------------------------------
# Fit smoke (PM3 - loss binding)
# ---------------------------------------------------------------------

class TestFitSmoke:

    def test_fit_one_step(self):
        m = _tiny_model()  # 2 levels => 32/4 ok
        m.compile(optimizer="adam", loss="mse")
        x = np.random.RandomState(4).randn(4, 32, 32, 3).astype("float32")
        y = {"reconstruction": x}
        hist = m.fit(x, y, epochs=1, batch_size=2, verbose=0)
        assert np.isfinite(hist.history["loss"][-1])
