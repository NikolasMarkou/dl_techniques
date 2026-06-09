"""Tests for DeepKernelPCA — the "working" bar.

This is a DEAD (imported-nowhere) layer implementing an idiosyncratic "deep"
kernel-PCA approximation. These tests cover the agreed working bar, NOT textbook
kernel-PCA correctness:

- init + config validation
- forward pass (training=False AND training=True), output shape == compute_output_shape
- graph-mode call (wrapped in a keras.Model) does not crash
- gradient flow: finite, non-None grads for on-path trainable weights
- .keras save/load round-trip reproduces outputs (atol 1e-6)
- get_config/from_config preserves constructor sentinels (components_per_level=None)
- get_explained_variance_ratio smoke (eager helper)
"""

import os
import numpy as np
import pytest
import keras
from keras import ops
import tensorflow as tf

from dl_techniques.layers.statistics.deep_kernel_pca import DeepKernelPCA

# sklearn is available in the venv (findings/pca-correctness.md ran a live probe).
from sklearn.datasets import make_circles
from sklearn.decomposition import KernelPCA
from sklearn.linear_model import LogisticRegression


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture
def input_dim() -> int:
    # NOTE: the layer reuses projection-weight rows as per-sample coefficients,
    # so feature_dim must be >= batch_size. Keep input_dim >= batch_size below.
    return 8


@pytest.fixture
def batch_size() -> int:
    return 6


@pytest.fixture
def sample_data(batch_size, input_dim) -> np.ndarray:
    rng = np.random.default_rng(1234)
    return rng.standard_normal((batch_size, input_dim)).astype("float32")


@pytest.fixture
def basic_layer() -> DeepKernelPCA:
    # Explicit components_per_level keeps each level's feature_dim >= batch_size.
    return DeepKernelPCA(
        num_levels=2,
        components_per_level=[6, 6],
        kernel_type="rbf",
    )


# ---------------------------------------------------------------------
# Init + config validation
# ---------------------------------------------------------------------

class TestDeepKernelPCA:

    def test_init_defaults(self):
        layer = DeepKernelPCA()
        assert layer.num_levels == 3
        assert layer.components_per_level is None
        assert layer._components_per_level_init is None
        assert layer.kernel_types == ["rbf", "rbf", "rbf"]
        assert not layer.built

    def test_init_explicit_components(self, basic_layer):
        assert basic_layer.num_levels == 2
        assert basic_layer.components_per_level == [6, 6]

    def test_init_invalid_num_levels(self):
        with pytest.raises(ValueError, match="num_levels must be positive"):
            DeepKernelPCA(num_levels=0)

    def test_init_invalid_coupling_strength(self):
        with pytest.raises(ValueError, match="coupling_strength"):
            DeepKernelPCA(coupling_strength=1.5)

    def test_init_invalid_regularization(self):
        with pytest.raises(ValueError, match="regularization_lambda"):
            DeepKernelPCA(regularization_lambda=-0.1)

    def test_init_invalid_kernel_type(self):
        with pytest.raises(ValueError, match="Invalid kernel type"):
            DeepKernelPCA(kernel_type="not_a_kernel")

    def test_init_kernel_type_list_length_mismatch(self):
        with pytest.raises(ValueError, match="must match num_levels"):
            DeepKernelPCA(num_levels=2, kernel_type=["rbf"])

    def test_per_level_kernel_params_not_aliased(self):
        # A single dict must be copied per level (not shared by reference).
        layer = DeepKernelPCA(num_levels=3, kernel_params={"gamma": 0.5})
        assert len(layer.kernel_params) == 3
        layer.kernel_params[0]["gamma"] = 99.0
        assert layer.kernel_params[1]["gamma"] == 0.5
        assert layer.kernel_params[2]["gamma"] == 0.5

    # -----------------------------------------------------------------
    # compute_output_shape gating
    # -----------------------------------------------------------------

    def test_compute_output_shape_before_build_raises(self):
        layer = DeepKernelPCA(num_levels=3, components_per_level=None)
        with pytest.raises(ValueError, match="requires the layer to be built"):
            layer.compute_output_shape((None, 8))

    def test_compute_output_shape_after_build(self, basic_layer, batch_size, input_dim):
        basic_layer.build((batch_size, input_dim))
        shape = basic_layer.compute_output_shape((batch_size, input_dim))
        assert shape == (batch_size, 12)  # 6 + 6

    # -----------------------------------------------------------------
    # Forward pass (eager) — training False and True
    # -----------------------------------------------------------------

    def test_forward_training_false(self, basic_layer, sample_data, batch_size):
        out = basic_layer(sample_data, training=False)
        assert out.shape == (batch_size, 12)
        assert np.all(np.isfinite(ops.convert_to_numpy(out)))

    def test_forward_training_true(self, basic_layer, sample_data, batch_size):
        out = basic_layer(sample_data, training=True)
        assert out.shape == (batch_size, 12)
        assert np.all(np.isfinite(ops.convert_to_numpy(out)))

    def test_output_shape_matches_compute_output_shape(self, basic_layer, sample_data, batch_size, input_dim):
        out = basic_layer(sample_data, training=False)
        expected = basic_layer.compute_output_shape((batch_size, input_dim))
        assert tuple(out.shape) == tuple(expected)

    def test_forward_adaptive_components(self):
        # components_per_level=None triggers the golden-ratio adaptive path.
        # input_dim must stay >= batch_size; use input_dim=10, batch=4.
        layer = DeepKernelPCA(num_levels=2, components_per_level=None, kernel_type="rbf")
        x = np.random.default_rng(0).standard_normal((4, 10)).astype("float32")
        out = layer(x, training=False)
        assert layer.components_per_level is not None
        assert tuple(out.shape) == (4, sum(layer.components_per_level))

    def test_single_level(self):
        layer = DeepKernelPCA(num_levels=1, components_per_level=[5])
        x = np.random.default_rng(0).standard_normal((4, 8)).astype("float32")
        out = layer(x, training=False)
        assert tuple(out.shape) == (4, 5)

    @pytest.mark.parametrize("kernel_type", ["rbf", "linear", "polynomial", "sigmoid", "cosine"])
    def test_forward_all_kernels(self, kernel_type, sample_data, batch_size):
        layer = DeepKernelPCA(num_levels=2, components_per_level=[6, 6], kernel_type=kernel_type)
        out = layer(sample_data, training=False)
        assert np.all(np.isfinite(ops.convert_to_numpy(out)))
        assert tuple(out.shape) == (batch_size, 12)

    def test_list_input_shape_build(self, batch_size, input_dim):
        # Functional API may pass a list of shapes to build.
        layer = DeepKernelPCA(num_levels=2, components_per_level=[6, 6])
        layer.build([(batch_size, input_dim)])
        assert layer.built

    # -----------------------------------------------------------------
    # Graph-mode (the key regression for l2_normalize / eye / slice / tanh)
    # -----------------------------------------------------------------

    def _make_model(self, input_dim, **kwargs):
        inp = keras.Input(shape=(input_dim,), batch_size=None)
        out = DeepKernelPCA(**kwargs)(inp)
        return keras.Model(inp, out)

    def test_graph_mode_model_call_training_false(self, sample_data, input_dim):
        model = self._make_model(input_dim, num_levels=2, components_per_level=[6, 6])
        out = model(sample_data, training=False)
        assert np.all(np.isfinite(ops.convert_to_numpy(out)))

    def test_graph_mode_model_call_training_true(self, sample_data, input_dim):
        model = self._make_model(input_dim, num_levels=2, components_per_level=[6, 6])
        out = model(sample_data, training=True)
        assert np.all(np.isfinite(ops.convert_to_numpy(out)))

    def test_tf_function_call(self, basic_layer, sample_data):
        @tf.function
        def run(x):
            return basic_layer(x, training=False)

        out = run(tf.convert_to_tensor(sample_data))
        assert np.all(np.isfinite(out.numpy()))

    def test_tf_function_call_training_true(self, basic_layer, sample_data):
        @tf.function
        def run(x):
            return basic_layer(x, training=True)

        out = run(tf.convert_to_tensor(sample_data))
        assert np.all(np.isfinite(out.numpy()))

    # -----------------------------------------------------------------
    # Gradient flow
    # -----------------------------------------------------------------

    def test_gradient_flow(self, basic_layer, sample_data):
        x = tf.convert_to_tensor(sample_data)
        with tf.GradientTape() as tape:
            out = basic_layer(x, training=True)
            loss = ops.mean(ops.square(out))
        grads = tape.gradient(loss, basic_layer.trainable_variables)
        assert len(basic_layer.trainable_variables) > 0
        # At least the on-path projection matrices must receive finite grads.
        on_path = [g for g in grads if g is not None]
        assert len(on_path) > 0
        for g in on_path:
            assert np.all(np.isfinite(ops.convert_to_numpy(g)))

    def test_projection_matrices_get_gradients(self, basic_layer, sample_data):
        x = tf.convert_to_tensor(sample_data)
        with tf.GradientTape() as tape:
            out = basic_layer(x, training=True)
            loss = ops.mean(ops.square(out))
        grads = tape.gradient(loss, basic_layer.projection_matrices)
        for g in grads:
            assert g is not None
            assert np.all(np.isfinite(ops.convert_to_numpy(g)))

    # -----------------------------------------------------------------
    # Serialization: config sentinels
    # -----------------------------------------------------------------

    def test_get_config_preserves_none_sentinel(self):
        layer = DeepKernelPCA(num_levels=3, components_per_level=None)
        # Force build (mutates self.components_per_level to a concrete list).
        layer.build((4, 10))
        assert layer.components_per_level is not None  # mutated post-build
        config = layer.get_config()
        # get_config must still report the ORIGINAL None sentinel.
        assert config["components_per_level"] is None
        rebuilt = DeepKernelPCA.from_config(config)
        assert rebuilt.components_per_level is None
        assert rebuilt._components_per_level_init is None
        assert rebuilt.num_levels == 3

    def test_get_config_preserves_explicit_components(self, basic_layer):
        basic_layer.build((6, 8))
        config = basic_layer.get_config()
        assert config["components_per_level"] == [6, 6]
        rebuilt = DeepKernelPCA.from_config(config)
        assert rebuilt.components_per_level == [6, 6]

    def test_get_config_preserves_kernel_type(self):
        layer = DeepKernelPCA(num_levels=2, components_per_level=[6, 6], kernel_type="polynomial")
        config = layer.get_config()
        assert config["kernel_type"] == "polynomial"
        rebuilt = DeepKernelPCA.from_config(config)
        assert rebuilt.kernel_types == ["polynomial", "polynomial"]

    def test_get_config_preserves_kernel_params(self):
        layer = DeepKernelPCA(num_levels=2, components_per_level=[6, 6], kernel_params={"gamma": 0.3})
        config = layer.get_config()
        assert config["kernel_params"] == {"gamma": 0.3}
        rebuilt = DeepKernelPCA.from_config(config)
        assert rebuilt.kernel_params[0]["gamma"] == 0.3

    # -----------------------------------------------------------------
    # .keras round-trip
    # -----------------------------------------------------------------

    def test_keras_round_trip(self, sample_data, input_dim, tmp_path):
        model = self._make_model(input_dim, num_levels=2, components_per_level=[6, 6])
        out_before = ops.convert_to_numpy(model(sample_data, training=False))

        path = os.path.join(str(tmp_path), "dkpca.keras")
        model.save(path)
        loaded = keras.models.load_model(path)
        out_after = ops.convert_to_numpy(loaded(sample_data, training=False))

        np.testing.assert_allclose(out_before, out_after, atol=1e-6)

    def test_keras_round_trip_adaptive(self, tmp_path):
        inp = keras.Input(shape=(10,))
        out = DeepKernelPCA(num_levels=2, components_per_level=None)(inp)
        model = keras.Model(inp, out)
        x = np.random.default_rng(7).standard_normal((4, 10)).astype("float32")
        out_before = ops.convert_to_numpy(model(x, training=False))

        path = os.path.join(str(tmp_path), "dkpca_adaptive.keras")
        model.save(path)
        loaded = keras.models.load_model(path)
        out_after = ops.convert_to_numpy(loaded(x, training=False))
        np.testing.assert_allclose(out_before, out_after, atol=1e-6)

    # -----------------------------------------------------------------
    # Public helper smoke
    # -----------------------------------------------------------------

    def test_explained_variance_ratio(self, basic_layer, sample_data):
        basic_layer(sample_data, training=False)
        ratios = basic_layer.get_explained_variance_ratio()
        assert len(ratios) == basic_layer.num_levels
        for r in ratios:
            assert np.all(np.isfinite(np.asarray(r)))

    def test_no_backward_coupling_attr(self, basic_layer):
        # The dead coupling_weights_backward list was removed.
        assert not hasattr(basic_layer, "coupling_weights_backward")

    def test_backward_coupling_disabled(self, sample_data, batch_size):
        layer = DeepKernelPCA(
            num_levels=2, components_per_level=[6, 6], use_backward_coupling=False
        )
        out = layer(sample_data, training=False)
        assert tuple(out.shape) == (batch_size, 12)


# ---------------------------------------------------------------------
# Textbook-correctness (genuine Nystrom kernel PCA via adapt())
# ---------------------------------------------------------------------
#
# Kernel-PCA components are sign- and rotation-ambiguous, so atol-vs-sklearn on
# raw components is INVALID. Correctness is established by class-separability on
# concentric circles, |corr| of the leading component vs sklearn, and
# eigenvalue-ordering -- NOT atol-vs-sklearn on raw components.
#
# The fitted path uses the canonical Nystrom out-of-sample formula (adapt =
# greedy layer-wise eigh of the centered training Gram; call = centered
# K(x, landmarks) @ alphas). With the exact RBF kernel this clears a HIGHER bar
# than the RFF-based ikPCA (>0.95 corr). gamma is tuned so the top-2 eigenvalues
# separate (avoids the near-degenerate rotation ambiguity).

# Tuned so leading eigenvalues separate (sep acc=1.0, |corr|~1.0 vs sklearn).
_GAMMA = 0.5


def _circles_embedded(input_dim: int = 20, seed: int = 0):
    """Concentric circles (2D) embedded into ``input_dim`` via a random map."""
    rng = np.random.default_rng(seed)
    x2, labels = make_circles(
        n_samples=300, factor=0.3, noise=0.05, random_state=seed
    )
    proj = rng.standard_normal((2, input_dim)).astype("float32")
    embedded = (x2.astype("float32") @ proj).astype("float32")
    return embedded, labels.astype("int32")


class TestDeepKernelPCACorrectness:

    def test_adapt_separates_concentric_circles(self):
        # After adapt, the top-k kernel-PCs linearly separate inner/outer.
        # Single genuine level is sufficient (multi-level greedy also passes;
        # see test_multilevel_greedy_separates). Nystrom has no batch<=feature
        # constraint, so input_dim is free.
        X, y = _circles_embedded()
        layer = DeepKernelPCA(
            num_levels=1, components_per_level=[2], kernel_type="rbf",
            kernel_params={"gamma": _GAMMA},
        )
        layer.adapt(X)
        comp = ops.convert_to_numpy(layer(X, training=False))
        acc = LogisticRegression(max_iter=1000).fit(comp, y).score(comp, y)
        assert acc > 0.90, f"fitted separability acc {acc:.3f} <= 0.90 (F1)"

    def test_corr_with_sklearn_kernelpca(self):
        # Leading fitted component vs sklearn KernelPCA leading component.
        X, _ = _circles_embedded()
        layer = DeepKernelPCA(
            num_levels=1, components_per_level=[2], kernel_type="rbf",
            kernel_params={"gamma": _GAMMA},
        )
        layer.adapt(X)
        comp = ops.convert_to_numpy(layer(X, training=False))

        sk = KernelPCA(n_components=2, kernel="rbf", gamma=_GAMMA).fit_transform(X)
        corr = abs(np.corrcoef(comp[:, 0], sk[:, 0])[0, 1])
        # Exact-RBF Nystrom -> higher bar than RFF. Do NOT lower below 0.90.
        assert corr > 0.90, f"|corr| {corr:.3f} <= 0.90 vs sklearn (F1)"

    def test_multilevel_greedy_separates(self):
        # Greedy multi-level stacking still separates the circles end-to-end.
        X, y = _circles_embedded()
        layer = DeepKernelPCA(
            num_levels=2, components_per_level=[4, 2], kernel_type="rbf",
            kernel_params={"gamma": _GAMMA},
        )
        layer.adapt(X)
        comp = ops.convert_to_numpy(layer(X, training=False))
        assert comp.shape == (X.shape[0], 6)
        acc = LogisticRegression(max_iter=1000).fit(comp, y).score(comp, y)
        assert acc > 0.90, f"multi-level greedy sep acc {acc:.3f} <= 0.90 (F1)"

    def test_adapt_generalizes_to_heldout(self):
        # OUT-OF-SAMPLE (canonical Nystrom generalization, WARNING#1):
        # adapt on a TRAIN split ONLY (these become the Nystrom landmarks),
        # then transform a HELD-OUT split and score a classifier trained on the
        # train components. This exercises the whole point of the Nystrom path:
        # projecting points NOT in the adapt set. The FITTED Nystrom path has NO
        # feature_dim >= batch_size constraint, so held-out scoring is fine.
        X, y = _circles_embedded()
        n = X.shape[0]
        n_tr = int(n * 0.70)
        X_tr, X_te = X[:n_tr], X[n_tr:]
        y_tr, y_te = y[:n_tr], y[n_tr:]
        layer = DeepKernelPCA(
            num_levels=1, components_per_level=[2], kernel_type="rbf",
            kernel_params={"gamma": _GAMMA},
        )
        layer.adapt(X_tr)  # fit on TRAIN ONLY (landmarks = train split)
        comp_tr = ops.convert_to_numpy(layer(X_tr, training=False))
        comp_te = ops.convert_to_numpy(layer(X_te, training=False))
        clf = LogisticRegression(max_iter=1000).fit(comp_tr, y_tr)
        acc = clf.score(comp_te, y_te)  # score on HELD-OUT
        assert acc > 0.90, f"held-out separability acc {acc:.3f} <= 0.90 (F1)"

    def test_unfitted_fails_corr_bar(self):
        # NEGATIVE CONTROL (WARNING#2): an UN-FITTED layer (no adapt = fitted
        # path inactive -> random-projection fallback) must FAIL the |corr| bar
        # the fitted layer clears at ~1.0. This pins that adapt() is what
        # establishes correctness, catching a silent adapt-no-op / broken
        # fitted-branch regression. The un-fitted fallback carries the legacy
        # feature_dim >= batch_size precondition, so use batch=64 with
        # components_per_level=[64] over input_dim=80 (feature_dim 80 >= 64).
        # Correlating over 64 points (not a tiny batch) keeps the control STABLE
        # (seeded; |corr|~0.48, well below 0.85; small-batch corr is noisy and
        # was deliberately avoided).
        keras.utils.set_random_seed(0)
        X, _ = _circles_embedded(input_dim=80)
        X_b = X[:64]
        layer = DeepKernelPCA(
            num_levels=1, components_per_level=[64], kernel_type="rbf",
            kernel_params={"gamma": _GAMMA},
        )
        comp = ops.convert_to_numpy(layer(X_b, training=False))  # NO adapt
        sk = KernelPCA(
            n_components=2, kernel="rbf", gamma=_GAMMA
        ).fit_transform(X_b)
        corr = abs(np.corrcoef(comp[:, 0], sk[:, 0])[0, 1])
        assert corr < 0.85, (
            f"un-fitted |corr| {corr:.3f} >= 0.85 -- negative control failed; "
            f"adapt() is not load-bearing"
        )

    def test_eigenvalues_sorted_descending(self):
        X, _ = _circles_embedded()
        layer = DeepKernelPCA(
            num_levels=2, components_per_level=[4, 2], kernel_type="rbf",
            kernel_params={"gamma": _GAMMA},
        )
        layer.adapt(X)
        for level in range(layer.num_levels):
            eig = ops.convert_to_numpy(layer.eigenvalues[level])
            assert np.all(np.diff(eig) <= 1e-4), \
                f"level {level} eigenvalues not descending: {eig}"
        # adapt must move level-0 eigenvalues off their all-ones init.
        assert not np.allclose(ops.convert_to_numpy(layer.eigenvalues[0]), 1.0)

    def test_adapt_then_serialization_preserves_fit(self, tmp_path):
        X, _ = _circles_embedded()
        inp = keras.Input(shape=(X.shape[1],))
        layer = DeepKernelPCA(
            num_levels=2, components_per_level=[4, 2], kernel_type="rbf",
            kernel_params={"gamma": _GAMMA},
        )
        out = layer(inp)
        model = keras.Model(inp, out)
        layer.adapt(X)

        fitted_eig = ops.convert_to_numpy(layer.eigenvalues[0])
        y_before = ops.convert_to_numpy(model(X, training=False))

        path = os.path.join(str(tmp_path), "dkpca_fitted.keras")
        model.save(path)
        restored = keras.models.load_model(path)
        y_after = ops.convert_to_numpy(restored(X, training=False))

        # genuine fitted transform survives the round-trip (not reset to the
        # un-fitted random-projection path).
        np.testing.assert_allclose(y_before, y_after, atol=1e-5)
        restored_eig = ops.convert_to_numpy(restored.layers[-1].eigenvalues[0])
        np.testing.assert_allclose(restored_eig, fitted_eig, atol=1e-4)
        assert not np.allclose(restored_eig, 1.0)

    def test_unfitted_still_runs(self):
        # Un-adapted layer must still run call() (random-projection fallback);
        # output is meaningless until adapt, but must not error. The UN-FITTED
        # fallback still carries the legacy feature_dim >= batch_size precondition
        # (prior-plan D-005), so use a small batch (<= input_dim) here. The
        # FITTED Nystrom path has NO such constraint (it runs on n=300 above).
        X, _ = _circles_embedded()
        # The un-fitted fallback needs every level's component count >= batch
        # (reuses projection-weight rows as coefficients). Use batch 6 with
        # components [6, 6] (the existing working-bar fixture's shape).
        X_small = X[:6]
        layer = DeepKernelPCA(
            num_levels=2, components_per_level=[6, 6], kernel_type="rbf",
            kernel_params={"gamma": _GAMMA},
        )
        out = ops.convert_to_numpy(layer(X_small, training=False))
        assert out.shape == (X_small.shape[0], 12)
        assert np.all(np.isfinite(out))
        # un-fitted eigenvalues remain at the all-ones init.
        assert np.allclose(ops.convert_to_numpy(layer.eigenvalues[0]), 1.0)

    def test_adapt_undersampled_raises(self):
        # n_samples - 1 < a level's component count -> clear ValueError.
        layer = DeepKernelPCA(
            num_levels=1, components_per_level=[8], kernel_type="rbf",
            kernel_params={"gamma": _GAMMA},
        )
        tiny = np.random.default_rng(0).standard_normal((4, 20)).astype("float32")
        with pytest.raises(ValueError, match="at least"):
            layer.adapt(tiny)

    def test_fitted_call_graph_mode(self):
        # Fitted Nystrom path must run under graph mode (tf.function / Model).
        X, _ = _circles_embedded()
        inp = keras.Input(shape=(X.shape[1],))
        layer = DeepKernelPCA(
            num_levels=2, components_per_level=[4, 2], kernel_type="rbf",
            kernel_params={"gamma": _GAMMA},
        )
        out = layer(inp)
        model = keras.Model(inp, out)
        layer.adapt(X)

        @tf.function
        def run(x):
            return model(x, training=False)

        y = run(tf.convert_to_tensor(X))
        assert np.all(np.isfinite(y.numpy()))
        assert tuple(y.shape) == (X.shape[0], 6)
