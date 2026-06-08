"""Tests for InvertibleKernelPCA + InvertibleKernelPCADenoiser — the "working" bar.

These are DEAD (imported-nowhere) layers implementing an idiosyncratic
RFF-based kernel-PCA approximation. These tests cover the agreed working bar,
NOT textbook kernel-PCA / analytic pre-image correctness:

- init + config validation
- forward pass (training=False AND training=True), output shape == compute_output_shape
- graph-mode call (wrapped in a keras.Model / tf.function) does not crash
  (regression for the old ops.nn.l2_normalize / in-call `.assign` / ops.median crashes)
- gradient flow: finite, non-None grads for on-path trainable weights
- .keras save/load round-trip reproduces outputs (atol 1e-6) for BOTH classes
  (the Denoiser round-trip exercises the new from_config + nested ikpca config)
- inverse_transform self-consistency / shape (documented LEARNED approximation,
  NOT analytic pre-image)
- get_config/from_config preserves constructor sentinels (gamma=None,
  n_components=None / float)
"""

import os
import numpy as np
import pytest
import keras
from keras import ops
import tensorflow as tf

from dl_techniques.layers.statistics.invertible_kernel_pca import (
    InvertibleKernelPCA,
    InvertibleKernelPCADenoiser,
)


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture
def input_dim() -> int:
    return 8


@pytest.fixture
def batch_size() -> int:
    return 6


@pytest.fixture
def sample_data(batch_size, input_dim) -> np.ndarray:
    rng = np.random.default_rng(1234)
    return rng.standard_normal((batch_size, input_dim)).astype("float32")


@pytest.fixture
def basic_layer() -> InvertibleKernelPCA:
    return InvertibleKernelPCA(
        n_components=4,
        n_random_features=16,
        random_seed=0,
    )


@pytest.fixture
def basic_denoiser() -> InvertibleKernelPCADenoiser:
    return InvertibleKernelPCADenoiser(
        n_components=4,
        n_random_features=16,
        kernel_type="rbf",
    )


# =====================================================================
# InvertibleKernelPCA
# =====================================================================

class TestInvertibleKernelPCA:

    # ---- init + config validation ----

    def test_init_defaults(self):
        layer = InvertibleKernelPCA()
        assert layer.n_components is None
        assert layer._n_components_init is None
        assert layer.gamma is None
        assert layer._gamma_init is None
        assert layer.n_random_features == 256
        assert layer.kernel_type == "rbf"
        assert not layer.built

    def test_init_explicit(self, basic_layer):
        assert basic_layer.n_components == 4
        assert basic_layer.n_random_features == 16

    def test_init_invalid_n_components(self):
        with pytest.raises(ValueError, match="n_components must be positive"):
            InvertibleKernelPCA(n_components=0)

    def test_init_invalid_n_random_features(self):
        with pytest.raises(ValueError, match="n_random_features must be positive"):
            InvertibleKernelPCA(n_random_features=0)

    def test_init_invalid_kernel_type(self):
        with pytest.raises(ValueError, match="kernel_type"):
            InvertibleKernelPCA(kernel_type="not_a_kernel")

    def test_init_invalid_regularization(self):
        with pytest.raises(ValueError, match="regularization"):
            InvertibleKernelPCA(regularization=-1.0)

    def test_build_components_exceed_features(self, input_dim):
        layer = InvertibleKernelPCA(n_components=32, n_random_features=16)
        with pytest.raises(ValueError, match="cannot be larger"):
            layer.build((None, input_dim))

    @pytest.mark.parametrize("kernel_type", ["rbf", "laplacian", "cauchy"])
    def test_build_all_kernels(self, kernel_type, input_dim):
        layer = InvertibleKernelPCA(
            n_components=4, n_random_features=16, kernel_type=kernel_type
        )
        layer.build((None, input_dim))
        assert layer.built
        assert layer.frequencies.shape == (input_dim, 16)

    # ---- forward pass ----

    def test_forward_inference(self, basic_layer, sample_data, batch_size):
        out = basic_layer(sample_data, training=False)
        assert tuple(out.shape) == (batch_size, 4)

    def test_forward_training(self, basic_layer, sample_data, batch_size):
        # training=True must NOT crash (regression: in-call `.assign` removed).
        out = basic_layer(sample_data, training=True)
        assert tuple(out.shape) == (batch_size, 4)

    def test_output_shape_matches_compute(self, basic_layer, sample_data):
        out = basic_layer(sample_data, training=False)
        expected = basic_layer.compute_output_shape(sample_data.shape)
        assert tuple(out.shape) == tuple(expected)

    def test_forward_finite(self, basic_layer, sample_data):
        out = ops.convert_to_numpy(basic_layer(sample_data, training=False))
        assert np.all(np.isfinite(out))

    def test_whiten_forward(self, sample_data, batch_size):
        layer = InvertibleKernelPCA(
            n_components=4, n_random_features=16, whiten=True, random_seed=0
        )
        out = ops.convert_to_numpy(layer(sample_data, training=False))
        assert out.shape == (batch_size, 4)
        assert np.all(np.isfinite(out))

    def test_no_center_features(self, sample_data, batch_size):
        layer = InvertibleKernelPCA(
            n_components=4, n_random_features=16,
            center_features=False, random_seed=0,
        )
        out = ops.convert_to_numpy(layer(sample_data, training=False))
        assert out.shape == (batch_size, 4)
        assert np.all(np.isfinite(out))

    # ---- graph mode ----

    def test_graph_mode_model(self, sample_data, input_dim, batch_size):
        # Regression for l2_normalize / .assign / graph-mode crashes.
        inp = keras.Input(shape=(input_dim,))
        layer = InvertibleKernelPCA(n_components=4, n_random_features=16, random_seed=1)
        model = keras.Model(inp, layer(inp))
        out = ops.convert_to_numpy(model(sample_data))
        assert out.shape == (batch_size, 4)
        assert np.all(np.isfinite(out))

    def test_graph_mode_tf_function_training(self, sample_data, basic_layer):
        _ = basic_layer(sample_data)  # build

        @tf.function
        def run(x):
            return basic_layer(x, training=True)

        out = ops.convert_to_numpy(run(ops.convert_to_tensor(sample_data)))
        assert np.all(np.isfinite(out))

    # ---- gradient flow ----

    def test_gradient_flow(self, basic_layer, sample_data):
        x = ops.convert_to_tensor(sample_data)
        _ = basic_layer(x)  # build
        with tf.GradientTape() as tape:
            comp = basic_layer(x, training=True)
            rec = basic_layer.inverse_transform(comp)
            # combine forward + inverse so every trainable weight is on-path
            loss = ops.mean(ops.square(rec)) + ops.mean(ops.square(comp))
        grads = tape.gradient(loss, basic_layer.trainable_variables)
        assert len(grads) > 0
        for v, g in zip(basic_layer.trainable_variables, grads):
            assert g is not None, f"None grad for {v.name}"
            assert np.all(np.isfinite(ops.convert_to_numpy(g))), f"non-finite grad {v.name}"

    # ---- inverse_transform ----

    def test_inverse_transform_shape(self, basic_layer, sample_data, batch_size, input_dim):
        comp = basic_layer(sample_data, training=False)
        rec = basic_layer.inverse_transform(comp)
        assert tuple(rec.shape) == (batch_size, input_dim)

    def test_inverse_transform_finite(self, basic_layer, sample_data):
        comp = basic_layer(sample_data, training=False)
        rec = ops.convert_to_numpy(basic_layer.inverse_transform(comp))
        assert np.all(np.isfinite(rec))

    def test_inverse_transform_self_consistent(self, basic_layer, sample_data):
        # LEARNED approximation (not analytic pre-image): we only assert that
        # the same components map deterministically to the same reconstruction.
        comp = basic_layer(sample_data, training=False)
        r1 = ops.convert_to_numpy(basic_layer.inverse_transform(comp))
        r2 = ops.convert_to_numpy(basic_layer.inverse_transform(comp))
        np.testing.assert_allclose(r1, r2, atol=1e-6)

    def test_reconstruction_error_smoke(self, basic_layer, sample_data, batch_size):
        # transform()/compute_reconstruction_error call self.call directly
        # (bypassing __call__), so the layer must be built first.
        _ = basic_layer(sample_data)
        err = ops.convert_to_numpy(basic_layer.compute_reconstruction_error(sample_data))
        assert err.shape == (batch_size,)
        assert np.all(np.isfinite(err))
        assert np.all(err >= 0.0)

    # ---- get_config / from_config sentinels ----

    def test_config_sentinels_none(self):
        # gamma=None, n_components=None must survive get_config even after build.
        layer = InvertibleKernelPCA(n_components=None, gamma=None, n_random_features=16)
        layer.build((None, 8))
        # build mutated the live attrs ...
        assert layer.gamma is not None
        assert layer.n_components is not None
        # ... but the serialized sentinels are still None.
        cfg = layer.get_config()
        assert cfg["n_components"] is None
        assert cfg["gamma"] is None
        rebuilt = InvertibleKernelPCA.from_config(cfg)
        assert rebuilt.n_components is None
        assert rebuilt.gamma is None
        assert not rebuilt.built

    def test_config_roundtrip_explicit(self, basic_layer):
        cfg = basic_layer.get_config()
        rebuilt = InvertibleKernelPCA.from_config(cfg)
        assert rebuilt.n_components == 4
        assert rebuilt.n_random_features == 16

    # ---- .keras round-trip ----

    def test_keras_roundtrip(self, sample_data, input_dim, tmp_path):
        inp = keras.Input(shape=(input_dim,))
        layer = InvertibleKernelPCA(n_components=4, n_random_features=16, random_seed=7)
        model = keras.Model(inp, layer(inp))

        y_before = ops.convert_to_numpy(model(sample_data))

        path = os.path.join(tmp_path, "ikpca.keras")
        model.save(path)
        restored = keras.models.load_model(path)
        y_after = ops.convert_to_numpy(restored(sample_data))

        np.testing.assert_allclose(y_before, y_after, atol=1e-6)


# =====================================================================
# InvertibleKernelPCADenoiser
# =====================================================================

class TestInvertibleKernelPCADenoiser:

    # ---- init + config validation ----

    def test_init_defaults(self):
        layer = InvertibleKernelPCADenoiser()
        assert layer.n_components_param == 0.95
        assert layer.n_components is None  # float -> resolved in build
        assert layer.variance_threshold == 0.95
        assert layer.ikpca is None
        assert not layer.built

    def test_init_int_components(self, basic_denoiser):
        assert basic_denoiser.n_components == 4
        assert basic_denoiser.variance_threshold is None
        assert basic_denoiser.n_components_param == 4

    def test_init_invalid_float_components(self):
        with pytest.raises(ValueError, match=r"must be in \(0, 1\]"):
            InvertibleKernelPCADenoiser(n_components=1.5)

    # ---- build ----

    def test_build_creates_child(self, basic_denoiser, input_dim):
        basic_denoiser.build((None, input_dim))
        assert basic_denoiser.built
        assert basic_denoiser.ikpca is not None
        assert basic_denoiser.ikpca.built

    def test_build_float_components(self, input_dim):
        layer = InvertibleKernelPCADenoiser(n_components=0.5, n_random_features=16)
        layer.build((None, input_dim))
        assert layer.n_components == max(1, int(16 * 0.5))
        assert layer.ikpca is not None

    # ---- forward ----

    def test_forward_inference(self, basic_denoiser, sample_data, batch_size, input_dim):
        out = basic_denoiser(sample_data, training=False)
        assert tuple(out.shape) == (batch_size, input_dim)

    def test_forward_training(self, basic_denoiser, sample_data, batch_size, input_dim):
        out = basic_denoiser(sample_data, training=True)
        assert tuple(out.shape) == (batch_size, input_dim)

    def test_forward_finite(self, basic_denoiser, sample_data):
        out = ops.convert_to_numpy(basic_denoiser(sample_data, training=False))
        assert np.all(np.isfinite(out))

    def test_adaptive_components(self, sample_data, batch_size, input_dim):
        layer = InvertibleKernelPCADenoiser(
            n_components=4, n_random_features=16, adaptive_components=True,
        )
        out = ops.convert_to_numpy(layer(sample_data, training=True))
        assert out.shape == (batch_size, input_dim)
        assert np.all(np.isfinite(out))

    @pytest.mark.parametrize("noise_estimation", ["mad", "std"])
    def test_estimate_noise_level(self, noise_estimation, sample_data, batch_size):
        # 'mad' exercises ops.median (regression: must not crash in this venv).
        layer = InvertibleKernelPCADenoiser(
            n_components=4, n_random_features=16, noise_estimation=noise_estimation,
        )
        layer.build((None, sample_data.shape[-1]))
        nl = ops.convert_to_numpy(layer.estimate_noise_level(ops.convert_to_tensor(sample_data)))
        assert nl.shape == (batch_size, 1)
        assert np.all(np.isfinite(nl))

    # ---- graph mode ----

    def test_graph_mode_model(self, sample_data, input_dim, batch_size):
        inp = keras.Input(shape=(input_dim,))
        layer = InvertibleKernelPCADenoiser(n_components=4, n_random_features=16)
        model = keras.Model(inp, layer(inp))
        out = ops.convert_to_numpy(model(sample_data))
        assert out.shape == (batch_size, input_dim)
        assert np.all(np.isfinite(out))

    def test_graph_mode_mad_tf_function(self, sample_data, input_dim):
        # ops.median inside estimate_noise_level under tf.function (regression).
        layer = InvertibleKernelPCADenoiser(
            n_components=4, n_random_features=16, noise_estimation="mad",
        )
        layer.build((None, input_dim))

        @tf.function
        def run(x):
            return layer.estimate_noise_level(x)

        out = ops.convert_to_numpy(run(ops.convert_to_tensor(sample_data)))
        assert np.all(np.isfinite(out))

    # ---- gradient flow ----

    def test_gradient_flow(self, basic_denoiser, sample_data):
        x = ops.convert_to_tensor(sample_data)
        _ = basic_denoiser(x)  # build
        with tf.GradientTape() as tape:
            out = basic_denoiser(x, training=True)
            loss = ops.mean(ops.square(out - x))
        grads = tape.gradient(loss, basic_denoiser.trainable_variables)
        assert len(grads) > 0
        # at least one on-path weight has a finite grad
        finite = [
            g is not None and np.all(np.isfinite(ops.convert_to_numpy(g)))
            for g in grads
        ]
        assert any(finite)

    # ---- config + .keras round-trip (from_config / nested ikpca) ----

    def test_get_config_nested_child(self, basic_denoiser, input_dim):
        basic_denoiser.build((None, input_dim))
        cfg = basic_denoiser.get_config()
        assert cfg["n_components"] == 4
        assert cfg["ikpca_config"] is not None

    def test_from_config_rebuilds_child(self, basic_denoiser, sample_data, input_dim):
        _ = basic_denoiser(sample_data)  # build + create child
        cfg = basic_denoiser.get_config()
        rebuilt = InvertibleKernelPCADenoiser.from_config(cfg)
        assert rebuilt.ikpca is not None
        rebuilt.build((None, input_dim))
        out = ops.convert_to_numpy(rebuilt(sample_data))
        assert out.shape == sample_data.shape
        assert np.all(np.isfinite(out))

    def test_keras_roundtrip(self, sample_data, input_dim, tmp_path):
        inp = keras.Input(shape=(input_dim,))
        layer = InvertibleKernelPCADenoiser(n_components=4, n_random_features=16)
        model = keras.Model(inp, layer(inp))

        y_before = ops.convert_to_numpy(model(sample_data))

        path = os.path.join(tmp_path, "ikpca_denoiser.keras")
        model.save(path)
        restored = keras.models.load_model(path)
        y_after = ops.convert_to_numpy(restored(sample_data))

        np.testing.assert_allclose(y_before, y_after, atol=1e-6)
