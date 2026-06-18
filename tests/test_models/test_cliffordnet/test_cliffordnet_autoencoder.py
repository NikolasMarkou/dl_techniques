"""
Test suite for the explicit-Laplacian-pyramid CliffordNet autoencoder
(autoencoder.py).

This file covers the ``CliffordLaplacianUNet`` model: config round-trip,
forward pass, ``.keras`` save/load, variants/factory, and fit-smoke. The
standalone ``LaplacianPyramidLevel`` helper now lives in
``dl_techniques.layers.laplacian_filter`` and its tests live in
``tests/test_layers/test_laplacian_filter.py``.
"""

import os

import keras
import pytest
import numpy as np

from dl_techniques.models.cliffordnet.autoencoder import (
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

    def test_keras_roundtrip_with_objects(self, tmp_path):
        m = CliffordLaplacianUNet(
            in_channels=3, level_channels=(8, 16, 32), level_blocks=(1, 1, 1),
            kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02),
            kernel_regularizer=keras.regularizers.L2(1e-4),
        )
        x = np.random.RandomState(7).randn(2, 64, 64, 3).astype("float32")
        y1 = keras.ops.convert_to_numpy(m(x)["reconstruction"])
        cfg1 = m.get_config()
        path = os.path.join(str(tmp_path), "m_obj.keras")
        m.save(path)
        m2 = keras.models.load_model(path)
        y2 = keras.ops.convert_to_numpy(m2(x)["reconstruction"])
        np.testing.assert_allclose(
            y1, y2, atol=1e-4,
            err_msg="object-initializer/regularizer round-trip drift",
        )
        cfg2 = m2.get_config()
        assert cfg2["kernel_initializer"] == cfg1["kernel_initializer"], \
            "initializer config not preserved"
        assert cfg2["kernel_regularizer"] == cfg1["kernel_regularizer"], \
            "regularizer config not preserved"


# ---------------------------------------------------------------------
# encode() bottleneck accessor (Step 1/2)
# ---------------------------------------------------------------------

class TestEncode:
    """Cover the public ``encode()`` bottleneck accessor."""

    def _assert_bottleneck(self, m, x):
        """encode(x) == deepest stage: spatial/2^(L-1), level_channels[-1]."""
        b, h, w = x.shape[0], x.shape[1], x.shape[2]
        div = 2 ** (m.num_levels - 1)
        z = m.encode(x)
        assert tuple(z.shape) == (
            b, h // div, w // div, m.level_channels[-1],
        ), (
            f"encode() bottleneck shape mismatch: got {tuple(z.shape)}, "
            f"expected {(b, h // div, w // div, m.level_channels[-1])}"
        )

    def test_encode_bottleneck_shape_2level(self):
        # 2 levels => divisor 2; channels = level_channels[-1] = 16.
        m = _tiny_model()
        x = np.random.RandomState(10).randn(2, 32, 32, 3).astype("float32")
        self._assert_bottleneck(m, x)

    def test_encode_bottleneck_shape_3level(self):
        # 3 levels => divisor 4; channels = level_channels[-1] = 32.
        m = _tiny_model(level_channels=(8, 16, 32), level_blocks=(1, 1, 1))
        x = np.random.RandomState(11).randn(2, 64, 64, 3).astype("float32")
        self._assert_bottleneck(m, x)

    def test_encode_bottleneck_shape_small_variant(self):
        # `small` is a 3-level variant => divisor 4.
        m = create_clifford_laplacian_unet("small")
        x = np.random.RandomState(12).randn(1, 64, 64, 3).astype("float32")
        self._assert_bottleneck(m, x)

    def test_encode_keras_roundtrip(self, tmp_path):
        """encode() AND reconstruction must match after .keras round-trip."""
        m = _tiny_model(level_channels=(8, 16, 32), level_blocks=(1, 1, 1))
        x = np.random.RandomState(13).randn(2, 64, 64, 3).astype("float32")
        recon1 = keras.ops.convert_to_numpy(m(x)["reconstruction"])
        z1 = keras.ops.convert_to_numpy(m.encode(x))

        path = os.path.join(str(tmp_path), "m.keras")
        m.save(path)
        m2 = keras.models.load_model(path)

        recon2 = keras.ops.convert_to_numpy(m2(x)["reconstruction"])
        z2 = keras.ops.convert_to_numpy(m2.encode(x))

        np.testing.assert_allclose(
            recon1, recon2, atol=1e-4,
            err_msg="reconstruction differs after .keras round-trip",
        )
        np.testing.assert_allclose(
            z1, z2, atol=1e-4,
            err_msg="encode() bottleneck differs after .keras round-trip",
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


# ---------------------------------------------------------------------
# Frozen Gabor stem
# ---------------------------------------------------------------------

class TestGaborStem:

    def test_stem_present_and_frozen_by_default(self):
        import keras
        m = _tiny_model()  # use_gabor_stem defaults True; 96 filters, 7x7
        assert m.gabor_stem is not None
        # Depthwise: applied per channel, no cross-channel mixing.
        assert isinstance(m.gabor_stem, keras.layers.DepthwiseConv2D)
        assert m.gabor_stem.trainable is False

        x = np.random.RandomState(0).randn(2, 32, 32, 3).astype("float32")
        out = m(x)  # builds the stem; spatial resolution preserved
        assert tuple(out["reconstruction"].shape) == (2, 32, 32, 3)
        # Depthwise kernel: (kh, kw, in_channels, depth_multiplier=filters).
        assert tuple(m.gabor_stem.kernel.shape) == (7, 7, 3, 96)
        # Per-channel output = in_channels * filters = 3 * 96 = 288.
        assert m.gabor_stem.compute_output_shape((None, 32, 32, 3))[-1] == 3 * 96

        # Frozen: stem weights must not appear in trainable_variables.
        stem_paths = {w.path for w in m.gabor_stem.weights}
        trainable_paths = {v.path for v in m.trainable_variables}
        assert not (stem_paths & trainable_paths)

    def test_stem_disabled(self):
        m = _tiny_model(use_gabor_stem=False)
        assert m.gabor_stem is None
        x = np.random.RandomState(1).randn(1, 32, 32, 3).astype("float32")
        assert tuple(m(x)["reconstruction"].shape) == (1, 32, 32, 3)

    def test_stem_custom_config_roundtrip(self):
        m = _tiny_model(gabor_filters=32, gabor_kernel_size=5)
        cfg = m.get_config()
        assert cfg["use_gabor_stem"] is True
        assert cfg["gabor_filters"] == 32
        assert cfg["gabor_kernel_size"] == 5

        m2 = CliffordLaplacianUNet.from_config(cfg)
        assert m2.gabor_stem is not None
        x = np.random.RandomState(5).randn(1, 32, 32, 3).astype("float32")
        m2(x)  # build
        assert tuple(m2.gabor_stem.kernel.shape) == (5, 5, 3, 32)
        # Per-channel output = in_channels * filters = 3 * 32 = 96.
        assert m2.gabor_stem.compute_output_shape((None, 32, 32, 3))[-1] == 3 * 32
        assert m2.gabor_stem.trainable is False


# ---------------------------------------------------------------------
# Output activation
# ---------------------------------------------------------------------

class TestOutputActivation:

    def test_default_is_linear(self):
        m = _tiny_model()
        assert m.output_activation is None
        assert m.get_config()["output_activation"] is None

    def test_tanh_bounds_output_and_roundtrips(self, tmp_path):
        m = _tiny_model(output_activation="tanh")
        # Large-magnitude input to push a linear head out of [-1, 1].
        x = np.random.RandomState(8).randn(2, 32, 32, 3).astype("float32") * 5.0
        y = keras.ops.convert_to_numpy(m(x)["reconstruction"])
        assert y.min() >= -1.0 and y.max() <= 1.0  # tanh-bounded

        cfg = m.get_config()
        assert cfg["output_activation"] == "tanh"

        # full .keras round-trip with the activation in the head
        path = os.path.join(str(tmp_path), "m_act.keras")
        m.save(path)
        m2 = keras.models.load_model(path)
        y2 = keras.ops.convert_to_numpy(m2(x)["reconstruction"])
        np.testing.assert_allclose(y, y2, atol=1e-4)
