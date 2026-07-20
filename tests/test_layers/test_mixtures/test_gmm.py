"""
Test suite for GMMLayer (differentiable Gaussian Mixture Model layer).

Covers initialization (incl. the default 'orthonormal' regression for D-001),
build/shape, both output modes, properties, serialization round-trip, the
isometric-kernel add_loss, and a training smoke test. Backend-agnostic
(keras.ops only) per repo convention.
"""

import os
import tempfile
import pytest
import numpy as np
import keras
from keras import ops
from typing import Dict, Any

from dl_techniques.layers.mixtures.gmm import GMMLayer


# --------------------------------------------------------------------- fixtures

@pytest.fixture
def random_seed() -> int:
    return 42


@pytest.fixture
def basic_config() -> Dict[str, Any]:
    """Basic config using a standard initializer for general tests."""
    return {
        "n_components": 4,
        "temperature": 1.0,
        "isometric_regularizer_strength": 0.01,
        "output_mode": "assignments",
        "cluster_axis": -1,
        "mean_initializer": "glorot_normal",  # standard initializer for tests
    }


@pytest.fixture
def sample_data_2d() -> keras.KerasTensor:
    np.random.seed(42)
    data = np.random.normal(0, 1, (32, 64)).astype(np.float32)
    return keras.ops.convert_to_tensor(data)


@pytest.fixture
def sample_data_4d() -> keras.KerasTensor:
    np.random.seed(42)
    data = np.random.normal(0, 1, (8, 28, 28, 3)).astype(np.float32)
    return keras.ops.convert_to_tensor(data)


# --------------------------------------------------------------- initialization

class TestGMMLayerInitialization:

    def test_valid_initialization(self, basic_config: Dict[str, Any]) -> None:
        layer = GMMLayer(**basic_config)
        assert layer.n_components == 4
        assert layer.temperature == 1.0
        assert layer.isometric_regularizer_strength == 0.01
        assert layer.output_mode == "assignments"
        assert layer.built is False
        assert layer.means is None

    def test_default_orthonormal_initialization(self) -> None:
        """Regression (D-001): default 'orthonormal' must NOT raise in __init__."""
        layer = GMMLayer(n_components=4)
        # 'orthonormal' is kept as a string and resolved lazily in build()
        assert layer.mean_initializer == "orthonormal"
        assert layer.built is False

    @pytest.mark.parametrize("param,value,msg", [
        ("n_components", 0, "n_components must be a positive integer"),
        ("n_components", -3, "n_components must be a positive integer"),
        ("temperature", 0.0, "temperature must be positive"),
        ("temperature", -1.0, "temperature must be positive"),
        ("isometric_regularizer_strength", -0.1, "isometric_regularizer_strength must be non-negative"),
        ("variance_floor", 0.0, "variance_floor must be positive"),
        ("variance_floor", -1e-3, "variance_floor must be positive"),
        ("output_mode", "bogus", "output_mode must be"),
    ])
    def test_invalid_initialization(self, param: str, value: Any, msg: str) -> None:
        kwargs = {"n_components": 4, param: value}
        with pytest.raises(ValueError, match=msg):
            GMMLayer(**kwargs)


# ----------------------------------------------------------------- build/shapes

class TestGMMLayerShapes:

    def test_weight_shapes_after_build(self, basic_config: Dict[str, Any], sample_data_2d) -> None:
        layer = GMMLayer(**basic_config)
        _ = layer(sample_data_2d)  # triggers build
        assert layer.built is True
        feat = int(sample_data_2d.shape[-1])
        assert tuple(layer.means.shape) == (4, feat)
        assert tuple(layer.log_variances.shape) == (4, feat)
        assert tuple(layer.mixture_logits.shape) == (4,)

    def test_compute_output_shape_assignments(self, basic_config: Dict[str, Any]) -> None:
        layer = GMMLayer(**basic_config)
        assert layer.compute_output_shape((None, 64)) == (None, 4)

    def test_compute_output_shape_mixture(self) -> None:
        layer = GMMLayer(n_components=4, output_mode="mixture", mean_initializer="glorot_normal")
        assert layer.compute_output_shape((None, 64)) == (None, 64)


# ---------------------------------------------------------------- forward pass

class TestGMMLayerForwardPass:

    def test_assignments_output(self, basic_config: Dict[str, Any], sample_data_2d) -> None:
        layer = GMMLayer(**basic_config)
        out = layer(sample_data_2d)
        assert tuple(out.shape) == (32, 4)
        out_np = ops.convert_to_numpy(out)
        # responsibilities are a softmax -> rows sum to 1, all in [0, 1]
        np.testing.assert_allclose(out_np.sum(axis=-1), np.ones(32), rtol=1e-5, atol=1e-5)
        assert out_np.min() >= 0.0 and out_np.max() <= 1.0

    def test_mixture_output(self, sample_data_2d) -> None:
        layer = GMMLayer(n_components=4, output_mode="mixture", mean_initializer="glorot_normal")
        out = layer(sample_data_2d)
        assert tuple(out.shape) == tuple(sample_data_2d.shape)
        assert np.all(np.isfinite(ops.convert_to_numpy(out)))

    def test_4d_assignments(self, basic_config: Dict[str, Any], sample_data_4d) -> None:
        layer = GMMLayer(**basic_config)
        out = layer(sample_data_4d)
        # cluster_axis=-1 (3 channels) -> last dim becomes n_components
        assert tuple(out.shape) == (8, 28, 28, 4)

    def test_training_flag_both(self, basic_config: Dict[str, Any], sample_data_2d) -> None:
        layer = GMMLayer(**basic_config)
        o_train = layer(sample_data_2d, training=True)
        o_inf = layer(sample_data_2d, training=False)
        assert tuple(o_train.shape) == tuple(o_inf.shape) == (32, 4)


# ------------------------------------------------------------------- properties

class TestGMMLayerProperties:

    def test_properties_after_build(self, basic_config: Dict[str, Any], sample_data_2d) -> None:
        layer = GMMLayer(**basic_config)
        _ = layer(sample_data_2d)
        feat = int(sample_data_2d.shape[-1])
        assert tuple(layer.component_means.shape) == (4, feat)
        var = ops.convert_to_numpy(layer.component_variances)
        assert var.shape == (4, feat)
        assert np.all(var >= layer.variance_floor - 1e-9)  # floored
        w = ops.convert_to_numpy(layer.mixture_weights)
        np.testing.assert_allclose(w.sum(), 1.0, rtol=1e-6, atol=1e-6)

    def test_reset_parameters(self, basic_config: Dict[str, Any], sample_data_2d) -> None:
        layer = GMMLayer(**basic_config)
        _ = layer(sample_data_2d)
        layer.reset_parameters()
        # log_variances reset to zeros, mixture_logits to zeros (uniform)
        np.testing.assert_allclose(ops.convert_to_numpy(layer.log_variances),
                                   np.zeros_like(ops.convert_to_numpy(layer.log_variances)),
                                   atol=1e-7)


# ---------------------------------------------------------------- serialization

class TestGMMLayerSerialization:

    def test_get_config_keys(self, basic_config: Dict[str, Any]) -> None:
        layer = GMMLayer(**basic_config)
        config = layer.get_config()
        for key in ["n_components", "temperature", "isometric_regularizer_strength",
                    "variance_floor", "output_mode", "cluster_axis",
                    "mean_initializer", "log_variance_initializer",
                    "mean_regularizer", "random_seed"]:
            assert key in config

    def test_from_config_roundtrip(self, basic_config: Dict[str, Any]) -> None:
        layer = GMMLayer(**basic_config)
        config = layer.get_config()
        layer2 = GMMLayer.from_config(config)
        assert layer2.n_components == layer.n_components
        assert layer2.temperature == layer.temperature
        assert layer2.output_mode == layer.output_mode

    def _roundtrip_model(self, layer_kwargs: Dict[str, Any], sample) -> None:
        inp = keras.Input(shape=(sample.shape[-1],))
        out = GMMLayer(**layer_kwargs, name="gmm")(inp)
        model = keras.Model(inp, out)
        y0 = model(sample)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "gmm.keras")
            model.save(path)
            loaded = keras.models.load_model(path)
        np.testing.assert_allclose(
            ops.convert_to_numpy(y0),
            ops.convert_to_numpy(loaded(sample)),
            rtol=1e-6, atol=1e-6,
        )

    def test_model_save_load_glorot(self, sample_data_2d) -> None:
        """SC4: serialization round-trip with a standard initializer."""
        self._roundtrip_model(
            {"n_components": 4, "mean_initializer": "glorot_normal"}, sample_data_2d
        )

    def test_model_save_load_default_orthonormal(self, sample_data_2d) -> None:
        """SC4 + D-002: round-trip with the default 'orthonormal' initializer."""
        self._roundtrip_model({"n_components": 4}, sample_data_2d)

    def test_model_save_load_low_rank(self, sample_data_2d) -> None:
        """SC3: the guide-§8.2 acceptance bar for covariance_type='low_rank'.

        A genuine ``model.save()`` -> ``keras.models.load_model()`` cycle through a
        ``.keras`` archive, NOT an in-memory get_config/from_config round-trip. This is
        the ONLY test that can catch a key missing from ``from_config`` or a
        ``covariance_factors`` weight that fails to restore, because both surface as a
        silent numerical difference (or a load-time crash) rather than a config mismatch.
        """
        self._roundtrip_model(
            {
                "n_components": 4,
                "covariance_type": "low_rank",
                "covariance_rank": 3,
                "mean_initializer": "glorot_normal",
                "log_variance_initializer": "glorot_normal",
                "factor_initializer": "he_normal",  # non-default: must survive the trip
            },
            sample_data_2d,
        )


# -------------------------------------------------------------------- add_loss

class TestGMMLayerIntegration:

    def test_losses_non_empty_on_training(self, sample_data_2d) -> None:
        """SC5: isometric add_loss registered + finite when training=True."""
        inp = keras.Input(shape=(sample_data_2d.shape[-1],))
        out = GMMLayer(
            n_components=4,
            isometric_regularizer_strength=0.1,
            mean_initializer="glorot_normal",
            log_variance_initializer=keras.initializers.RandomNormal(stddev=1.0, seed=7),
            name="gmm",
        )(inp)
        model = keras.Model(inp, out)
        _ = model(sample_data_2d, training=True)
        assert len(model.losses) > 0
        loss_vals = [float(ops.convert_to_numpy(l)) for l in model.losses]
        assert all(np.isfinite(v) for v in loss_vals)
        # with random (anisotropic) log-variances the penalty is strictly positive
        assert sum(loss_vals) > 0.0

    def test_no_loss_when_strength_zero(self, sample_data_2d) -> None:
        inp = keras.Input(shape=(sample_data_2d.shape[-1],))
        out = GMMLayer(
            n_components=4,
            isometric_regularizer_strength=0.0,
            mean_initializer="glorot_normal",
            name="gmm",
        )(inp)
        model = keras.Model(inp, out)
        _ = model(sample_data_2d, training=True)
        assert len(model.losses) == 0

    def test_training_step_runs(self, sample_data_2d) -> None:
        """Backend-agnostic gradient-flow smoke: one fit step completes."""
        inp = keras.Input(shape=(sample_data_2d.shape[-1],))
        out = GMMLayer(n_components=4, mean_initializer="glorot_normal", name="gmm")(inp)
        model = keras.Model(inp, out)
        model.compile(optimizer="adam", loss="mse")
        target = np.random.RandomState(0).uniform(size=(32, 4)).astype(np.float32)
        history = model.fit(ops.convert_to_numpy(sample_data_2d), target,
                            epochs=1, batch_size=16, verbose=0)
        assert np.isfinite(history.history["loss"][-1])


# ------------------------------------------------------ posterior correctness

class TestGMMLayerPosteriorCorrectness:
    """Pin weights to known values and check responsibilities equal the analytic
    GMM posterior (closes review WARNING #1 — only sum-to-1 was asserted before)."""

    def test_responsibilities_match_analytic_posterior(self) -> None:
        import numpy as np
        # temperature=1, uniform mixing logits (0), unit variances (log_var=0):
        # responsibility_k = softmax_k(-0.5 * ||x - mu_k||^2)  (logdet=0, pi uniform,
        # D log 2pi is a per-row constant that cancels in the softmax).
        feat, k = 5, 3
        layer = GMMLayer(n_components=k, temperature=1.0, mean_initializer="glorot_normal")
        x = np.random.RandomState(0).normal(size=(7, feat)).astype("float32")
        _ = layer(x)  # build
        means = np.random.RandomState(1).normal(size=(k, feat)).astype("float32")
        layer.means.assign(means)  # log_variances=0, mixture_logits=0 from default init

        out = ops.convert_to_numpy(layer(x))

        # analytic
        sq = ((x[:, None, :] - means[None, :, :]) ** 2).sum(axis=-1)  # (7, k)
        logits = -0.5 * sq
        logits -= logits.max(axis=-1, keepdims=True)
        expected = np.exp(logits)
        expected /= expected.sum(axis=-1, keepdims=True)

        np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)

    def test_temperature_sharpens(self) -> None:
        import numpy as np
        x = np.random.RandomState(2).normal(size=(6, 4)).astype("float32")
        soft = GMMLayer(n_components=3, temperature=1.0, mean_initializer="glorot_normal")
        hard = GMMLayer(n_components=3, temperature=0.05, mean_initializer="glorot_normal")
        o_soft = ops.convert_to_numpy(soft(x))
        o_hard = ops.convert_to_numpy(hard(x))
        # lower temperature -> sharper (higher max responsibility on average)
        assert o_hard.max(axis=-1).mean() >= o_soft.max(axis=-1).mean()

    def test_mixture_mode_equals_responsibilities_dot_means(self) -> None:
        import numpy as np
        feat, k = 5, 3
        layer = GMMLayer(n_components=k, output_mode="mixture", mean_initializer="glorot_normal")
        x = np.random.RandomState(3).normal(size=(8, feat)).astype("float32")
        recon = ops.convert_to_numpy(layer(x))

        # recompute responsibilities from an assignments-mode twin sharing the weights
        means = ops.convert_to_numpy(layer.means)
        twin = GMMLayer(n_components=k, output_mode="assignments", mean_initializer="glorot_normal")
        _ = twin(x)
        twin.means.assign(means)
        twin.log_variances.assign(layer.log_variances)
        twin.mixture_logits.assign(layer.mixture_logits)
        resp = ops.convert_to_numpy(twin(x))

        np.testing.assert_allclose(recon, resp @ means, rtol=1e-5, atol=1e-5)


# --------------------------------------------------------------- graph mode

class TestGMMLayerGraphMode:
    """Graph-compatibility regression: call() must trace with a symbolic flag."""

    def test_graph_mode_symbolic_training(self, basic_config: Dict[str, Any]) -> None:
        """Regression for the bare ``if training and ...:`` graph-breaker."""
        import tensorflow as tf

        layer = GMMLayer(**basic_config)
        x = tf.constant(np.random.normal(0, 1, (8, 64)).astype(np.float32))

        @tf.function
        def run(inp, training):
            return layer(inp, training=training)

        y_train = run(x, tf.constant(True))
        y_infer = run(x, tf.constant(False))
        assert tuple(y_train.shape) == (8, 4)
        assert tuple(y_infer.shape) == (8, 4)

    def test_symbolic_training_fires_isometric_loss(self, basic_config: Dict[str, Any]) -> None:
        """A SYMBOLIC training=True tensor must fire the isometric add_loss (== the
        python-True value); symbolic False must contribute zero (the foot-gun fix).
        """
        import tensorflow as tf

        layer = GMMLayer(**basic_config)
        x = tf.constant(np.random.normal(0, 1, (16, 64)).astype(np.float32))
        layer.build((16, 64))
        # Perturb log-variances so covariances are anisotropic -> non-zero isometric loss
        # (a fresh isotropic init has a genuinely zero isometric penalty).
        layer.log_variances.assign(
            tf.constant(np.random.normal(0, 1, layer.log_variances.shape).astype(np.float32))
        )

        y = layer(x, training=True)
        python_loss = float(ops.convert_to_numpy(tf.add_n(layer.losses)))

        @tf.function
        def step(inp, training):
            _ = layer(inp, training=training)
            return tf.add_n(layer.losses) if layer.losses else tf.constant(0.0)

        sym_true = float(step(x, tf.constant(True)))
        sym_false = float(step(x, tf.constant(False)))
        assert python_loss > 0.0
        assert np.isclose(sym_true, python_loss, atol=1e-6), \
            "symbolic training=True isometric loss must equal the python-True value"
        assert sym_false == 0.0, "symbolic training=False must contribute zero loss"

    def test_compute_output_shape_before_build_multi_axis(self) -> None:
        """compute_output_shape correct PRE-build for multi/negative axes."""
        layer = GMMLayer(n_components=5, cluster_axis=[-2, -1], output_mode="assignments")
        assert layer.compute_output_shape((8, 3, 16)) == (8, 5)
        y = layer(np.random.normal(0, 1, (8, 3, 16)).astype(np.float32))
        assert tuple(y.shape) == (8, 5)

    @pytest.mark.parametrize("output_mode", ["assignments", "mixture"])
    def test_mixed_float16_forward(self, output_mode: str) -> None:
        """Forward must run under a mixed_float16 policy (float32 density math,
        compute_dtype output) without the autocast half-vs-float Sub crash.
        """
        original_policy = keras.mixed_precision.global_policy()
        try:
            keras.mixed_precision.set_global_policy("mixed_float16")
            layer = GMMLayer(
                n_components=4, output_mode=output_mode, mean_initializer="glorot_normal"
            )
            x = np.random.normal(0, 1, (8, 16)).astype(np.float32)
            y = layer(x)
            # Layer must emit the policy compute dtype and be finite.
            assert keras.backend.standardize_dtype(y.dtype) == "float16"
            y_np = np.asarray(ops.convert_to_numpy(y), dtype=np.float32)
            assert not np.isnan(y_np).any() and not np.isinf(y_np).any()
            # Internal parameters stay float32 (autocast=False).
            assert keras.backend.standardize_dtype(layer.means.dtype) == "float32"
        finally:
            keras.mixed_precision.set_global_policy(original_policy)


# ------------------------------------------------- low-rank covariance (oracle)

class TestGMMLayerLowRankCovariance:
    """Numerical gate for the low-rank-plus-diagonal covariance path.

    The Woodbury/Cholesky implementation is fast but opaque: a wrong batched
    ``solve_triangular`` convention does NOT raise, it silently returns a
    plausible-but-wrong log-density that still trains. The dense oracle below is
    the standing defense. If it fails, do NOT loosen its tolerance -- the
    implementation is wrong, not the test.
    """

    @staticmethod
    def _dense_reference_log_density(
        x: np.ndarray, means: np.ndarray, variances: np.ndarray, factors: np.ndarray
    ) -> np.ndarray:
        """Reference log-density built from an EXPLICIT dense Sigma_k.

        Deliberately naive: materializes ``Sigma_k = diag(d_k) + U_k U_k^T`` and uses
        a dense inverse and a dense log-determinant. It shares no code path, no
        identity, and no linear-algebra shortcut with the layer under test.

        :param x: Inputs, shape ``(batch, D)``.
        :param means: Component means, shape ``(K, D)``.
        :param variances: Floored diagonals ``d_k``, shape ``(K, D)``.
        :param factors: Low-rank factors ``U_k``, shape ``(K, D, R)``.
        :return: Log-densities, shape ``(batch, K)``.
        """
        n_components, feature_dims = means.shape
        out = np.empty((x.shape[0], n_components), dtype=np.float64)
        for k in range(n_components):
            sigma = np.diag(variances[k]) + factors[k] @ factors[k].T
            sigma_inv = np.linalg.inv(sigma)
            sign, logabsdet = np.linalg.slogdet(sigma)
            assert sign > 0, f"reference Sigma_{k} must be positive definite"
            diff = x - means[k]                                  # (batch, D)
            quad = np.einsum('bi,ij,bj->b', diff, sigma_inv, diff)
            out[:, k] = -0.5 * (
                quad + logabsdet + feature_dims * np.log(2.0 * np.pi)
            )
        return out

    @pytest.mark.parametrize("n_components,feature_dims,rank,batch", [
        (3, 6, 2, 16),    # original SC4 shape
        (2, 11, 4, 9),    # gap 1: odd D, larger R, K != R, batch != D
    ])
    def test_low_rank_matches_dense_reference(
        self, n_components: int, feature_dims: int, rank: int, batch: int
    ) -> None:
        """SC4 / A2: the Woodbury path must equal an explicit dense inverse + logdet.

        This is the gate for the whole low-rank feature.

        Two shapes, deliberately chosen so no two axis lengths coincide in the second
        case. A transposed or mis-broadcast axis in the batched ``solve_triangular``
        call can stay silent when ``K``, ``D``, ``R`` and ``batch`` happen to be
        compatible; ``(K=2, D=11, R=4, batch=9)`` shares no dimension with any other,
        so a swapped axis raises or diverges instead of quietly returning a plausible
        density.
        """
        layer = GMMLayer(
            n_components=n_components,
            covariance_type="low_rank",
            covariance_rank=rank,
            mean_initializer="glorot_normal",
            log_variance_initializer="glorot_normal",  # non-trivial, non-unit variances
            variance_floor=1e-3,
            random_seed=0,
        )
        np.random.seed(7)
        x = np.random.normal(0, 1, (batch, feature_dims)).astype(np.float32)
        layer(x)  # build

        assert layer.covariance_factors is not None
        assert tuple(layer.covariance_factors.shape) == (n_components, feature_dims, rank)

        # Read the ACTUAL parameters back out; the reference must not assume defaults.
        means = np.asarray(ops.convert_to_numpy(layer.means), dtype=np.float64)
        variances = np.asarray(
            ops.convert_to_numpy(layer.component_variances), dtype=np.float64
        )
        factors = np.asarray(
            ops.convert_to_numpy(layer.covariance_factors), dtype=np.float64
        )

        actual = np.asarray(
            ops.convert_to_numpy(layer._log_gaussian_density(ops.convert_to_tensor(x))),
            dtype=np.float64,
        )
        expected = self._dense_reference_log_density(
            x.astype(np.float64), means, variances, factors
        )

        max_deviation = np.abs(actual - expected).max()
        assert max_deviation <= 1e-5, (
            f"low-rank log-density deviates from the dense reference by "
            f"{max_deviation:.3e} (> 1e-5). Do NOT loosen this tolerance -- suspect "
            f"the batched solve_triangular RHS convention first."
        )


# --------------------------------------------- low-rank equivalence / compat
# Helpers shared by the equivalence tests below.

_SHARED_PARAM_NAMES = ("means", "log_variances", "mixture_logits")


def _copy_shared_parameters(source: GMMLayer, target: GMMLayer) -> None:
    """Copy the three covariance-type-independent weights from one layer to another.

    Both layers must already be built and agree on ``n_components`` / ``feature_dims``.
    ``covariance_factors`` is deliberately NOT copied -- it exists only under
    ``'low_rank'`` and is what the callers vary.

    :param source: Built layer to read parameters from.
    :type source: GMMLayer
    :param target: Built layer to write parameters into. Mutated in place.
    :type target: GMMLayer
    :return: None.
    :rtype: None
    """
    for name in _SHARED_PARAM_NAMES:
        getattr(target, name).assign(getattr(source, name))


class TestGMMLayerLowRankEquivalence:
    """SC5 / invariant I1: the low-rank path must not disturb the diagonal path.

    Two independent directions:

    * zero-``U`` collapses the Woodbury branch EXACTLY onto the diagonal branch, and
    * the default (diagonal) config still reproduces the closed-form diagonal posterior.
    """

    def test_zero_factors_equals_diagonal(self) -> None:
        """SC5 / 3b: with ``U = 0`` the Woodbury path collapses onto the diagonal path.

        With ``U = 0`` the capacitance matrix is exactly ``I_R``, so its Cholesky is
        ``I_R``, ``log det(M) = 0``, and the Woodbury correction is exactly zero. The two
        branches must therefore agree to floating-point noise -- not merely approximately.

        Zeroing ``U`` here is deliberate TEST INSTRUMENTATION, not a supported runtime
        configuration: ``dL/dU`` vanishes identically at ``U = 0``, which is exactly why
        ``factor_initializer`` defaults to 'glorot_uniform' (see D-005).
        """
        n_components, feature_dims, rank = 3, 6, 2
        x = np.random.RandomState(11).normal(
            size=(9, feature_dims)
        ).astype("float32")

        shared = {
            "n_components": n_components,
            "mean_initializer": "glorot_normal",
            "log_variance_initializer": "glorot_normal",  # anisotropic, not unit
            "temperature": 1.0,
        }
        diagonal = GMMLayer(**shared)
        low_rank = GMMLayer(
            covariance_type="low_rank", covariance_rank=rank, **shared
        )
        diagonal(x)
        low_rank(x)

        _copy_shared_parameters(diagonal, low_rank)
        low_rank.covariance_factors.assign(
            np.zeros((n_components, feature_dims, rank), dtype="float32")
        )

        np.testing.assert_allclose(
            ops.convert_to_numpy(low_rank(x)),
            ops.convert_to_numpy(diagonal(x)),
            atol=1e-6,
            err_msg=(
                "zero-U low-rank output diverged from the diagonal output; the Woodbury "
                "branch does not collapse to the diagonal branch as it must"
            ),
        )

    def test_diagonal_backward_compatibility(self) -> None:
        """SC5 / 3c: the DEFAULT config still equals the closed-form diagonal posterior.

        The reference is built here from explicitly seeded parameters and the plain
        diagonal-Gaussian formula -- i.e. the pre-change behavior, reconstructed
        independently rather than snapshotted. If adding ``covariance_type`` perturbed
        the default path in any way, this fails.
        """
        n_components, feature_dims, batch = 4, 5, 10
        rng = np.random.RandomState(23)
        x = rng.normal(size=(batch, feature_dims)).astype("float32")
        means = rng.normal(size=(n_components, feature_dims)).astype("float32")
        log_variances = rng.normal(
            scale=0.5, size=(n_components, feature_dims)
        ).astype("float32")
        mixture_logits = rng.normal(size=(n_components,)).astype("float32")

        # Default construction: covariance_type / covariance_rank / factor_initializer
        # are all left unspecified. No covariance_factors weight may appear.
        layer = GMMLayer(n_components=n_components, mean_initializer="glorot_normal")
        layer(x)
        assert layer.covariance_type == "diagonal"
        assert layer.covariance_factors is None, (
            "the default diagonal config must create NO covariance_factors weight "
            "(invariant I1: existing checkpoints stay weight-compatible)"
        )

        layer.means.assign(means)
        layer.log_variances.assign(log_variances)
        layer.mixture_logits.assign(mixture_logits)

        actual = np.asarray(ops.convert_to_numpy(layer(x)), dtype=np.float64)

        # Closed-form diagonal reference (float64, no layer code shared).
        variances = np.maximum(
            np.exp(log_variances.astype(np.float64)), layer.variance_floor
        )
        diff = x.astype(np.float64)[:, None, :] - means.astype(np.float64)[None, :, :]
        mahalanobis = (diff ** 2 / variances[None, :, :]).sum(axis=-1)
        log_det = np.log(variances).sum(axis=-1)
        log_density = -0.5 * (
            mahalanobis + log_det[None, :] + feature_dims * np.log(2.0 * np.pi)
        )
        logits = mixture_logits.astype(np.float64)
        log_mixing = logits - np.log(np.exp(logits).sum())
        joint = (log_density + log_mixing[None, :]) / layer.temperature
        joint -= joint.max(axis=-1, keepdims=True)
        expected = np.exp(joint)
        expected /= expected.sum(axis=-1, keepdims=True)

        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


# ------------------------------------------ low-rank gradients / config / notice

class TestGMMLayerLowRankTraining:
    """Gradient flow through ``covariance_factors``, plus the D-002 falsification check."""

    @staticmethod
    def _factor_gradient(strength: float) -> "np.ndarray":
        """Gradient of (MSE + registered layer losses) w.r.t. ``covariance_factors``.

        Both call sites must compare gradients at the SAME parameter point, so every
        weight is pinned to a fixed seeded value before the tape runs. Only
        ``isometric_regularizer_strength`` differs between calls.

        :param strength: Value for ``isometric_regularizer_strength``.
        :type strength: float
        :return: Gradient array of shape ``(K, D, R)``.
        :rtype: np.ndarray
        """
        import tensorflow as tf

        n_components, feature_dims, rank, batch = 3, 6, 2, 12
        rng = np.random.RandomState(5)
        x = rng.normal(size=(batch, feature_dims)).astype("float32")
        target = rng.uniform(size=(batch, n_components)).astype("float32")

        layer = GMMLayer(
            n_components=n_components,
            covariance_type="low_rank",
            covariance_rank=rank,
            isometric_regularizer_strength=strength,
            mean_initializer="glorot_normal",
        )
        layer(x)  # build

        # Pin ALL parameters so the two configurations are compared at one point.
        layer.means.assign(
            rng.normal(size=(n_components, feature_dims)).astype("float32")
        )
        layer.log_variances.assign(
            rng.normal(scale=0.5, size=(n_components, feature_dims)).astype("float32")
        )
        layer.mixture_logits.assign(rng.normal(size=(n_components,)).astype("float32"))
        layer.covariance_factors.assign(
            rng.normal(scale=0.3, size=(n_components, feature_dims, rank)).astype("float32")
        )

        with tf.GradientTape() as tape:
            out = layer(x, training=True)
            loss = tf.reduce_mean(tf.square(out - target))
            if layer.losses:
                loss = loss + tf.add_n(layer.losses)
        grad = tape.gradient(loss, layer.covariance_factors)
        assert grad is not None, (
            "covariance_factors received a None gradient -- the weight is disconnected "
            "from the loss and can never train"
        )
        return np.asarray(ops.convert_to_numpy(grad), dtype=np.float64)

    def test_gradient_flows_through_covariance_factors(self) -> None:
        """3d: ``covariance_factors`` gradients are non-None and non-zero."""
        grad = self._factor_gradient(strength=0.0)
        assert np.all(np.isfinite(grad))
        assert np.abs(grad).max() > 0.0, (
            "covariance_factors gradient is identically zero -- the low-rank factor is a "
            "dead weight (this is exactly what a zeros initializer would produce)"
        )

    def test_isometric_regularizer_does_not_suppress_factor_gradients(self) -> None:
        """Pre-Mortem #2 / D-002 falsification check -- NOT a routine assertion.

        D-002 chose Option D (reframe the isometric regularizer as a prior on the
        residual diagonal of the FACTOR-ANALYSIS form ``Sigma_k = diag(d_k) + U_k U_k^T``
        -- not PPCA, which would require an isotropic ``d_k`` this layer never enforces)
        over Option B (extend the loss to U's Gram spectrum). That
        choice is a falsifiable CLAIM: that the default-on regularizer complements the
        low-rank feature rather than fighting it. The pre-registered empirical form of
        "quietly fighting" is ``covariance_factors`` gradients suppressed by more than an
        order of magnitude versus a ``strength=0.0`` control.

        If this fires, do NOT tune the strength and do NOT loosen the bound. D-002 is
        falsified and must escalate to Option B.

        .. warning::
           **Read the measured ratio honestly.** The isometric loss reads
           ``log_variances`` only, so ``U`` is not on its computational graph at all and
           ``d(iso)/dU`` is a STRUCTURAL zero. This check therefore returns exactly 1.0
           by construction, and it can only ever fail if a future edit wires the
           regularizer to ``U``. That makes it a real guard against a real regression --
           but it is a SINGLE-POINT check, and it does not and cannot observe the
           INDIRECT effect: the regularizer shapes ``d_k``, and ``d_k`` enters ``U``'s
           gradient through the Woodbury terms, so two training TRAJECTORIES still
           diverge. Falsifying D-002 on the indirect channel would need a fitted-model
           comparison, which is out of this suite's scope.
        """
        regularized = self._factor_gradient(strength=0.01)  # the ctor default
        control = self._factor_gradient(strength=0.0)

        regularized_norm = float(np.linalg.norm(regularized))
        control_norm = float(np.linalg.norm(control))
        assert control_norm > 0.0, "control gradient is zero; the comparison is vacuous"

        ratio = regularized_norm / control_norm
        assert ratio > 0.1, (
            f"covariance_factors gradient norm fell to {ratio:.4f}x the "
            f"strength=0.0 control (< 0.1x = more than an order of magnitude). "
            f"Pre-Mortem #2 has FIRED: the isometric regularizer is quietly fighting the "
            f"low-rank feature, which falsifies D-002's factor-analysis reframe. Do NOT tune the "
            f"strength or relax this bound -- escalate D-002 to Option B (extend the "
            f"loss to U's Gram spectrum)."
        )

    def test_low_rank_config_roundtrip(self) -> None:
        """3d: the three new keys survive get_config/from_config.

        Per-key ``in`` / ``==`` checks, per repo norm -- never a full-dict equality,
        which would break on every unrelated Keras base-config addition.
        """
        layer = GMMLayer(
            n_components=4,
            covariance_type="low_rank",
            covariance_rank=3,
            factor_initializer="he_normal",
            mean_initializer="glorot_normal",
        )
        config = layer.get_config()

        for key in ("covariance_type", "covariance_rank", "factor_initializer"):
            assert key in config, f"get_config() is missing '{key}'"
        assert config["covariance_type"] == "low_rank"
        assert config["covariance_rank"] == 3

        restored = GMMLayer.from_config(config)
        assert restored.covariance_type == "low_rank"
        assert restored.covariance_rank == 3
        assert isinstance(restored.factor_initializer, keras.initializers.HeNormal)

    @pytest.mark.parametrize("covariance_type,should_fire", [
        ("low_rank", True),
        ("diagonal", False),
    ])
    def test_isometric_notice_fires_only_for_non_diagonal(
        self, caplog, covariance_type: str, should_fire: bool
    ) -> None:
        """SC9 / D-002: the reframe must be ANNOUNCED, not left to the docstring.

        Silence is the one outcome D-002 rules out. The notice must fire on
        ``low_rank`` + ``strength > 0`` and must stay quiet under ``diagonal``, where the
        regularizer's original semantics are unchanged and a notice would be noise.
        """
        import logging

        with caplog.at_level(logging.INFO, logger="dl"):
            GMMLayer(
                n_components=4,
                covariance_type=covariance_type,
                isometric_regularizer_strength=0.01,
                mean_initializer="glorot_normal",
            )

        fired = any(
            "isometric-kernel regularizer" in record.message
            for record in caplog.records
            if record.levelno == logging.INFO
        )
        assert fired is should_fire, (
            f"covariance_type='{covariance_type}': expected the D-002 notice "
            f"{'to fire' if should_fire else 'to stay silent'}, but it "
            f"{'fired' if fired else 'did not fire'}"
        )


# --------------------------------------------------------------- reset (D-005)

class TestGMMLayerLowRankReset:
    """D-005: ``reset_parameters()`` must be truthful under BOTH covariance modes."""

    def test_reset_reinitializes_covariance_factors(self) -> None:
        """3e: reset re-DRAWS ``covariance_factors`` -- it does not zero them.

        Two distinct claims, both load-bearing:

        1. the values actually CHANGE (before the fix they were silently retained, so a
           "reset" layer kept its old covariance structure), and
        2. they are NOT zeroed -- ``dL/dU`` vanishes identically at ``U = 0``, so zeroing
           would swap a stale weight for a permanently untrainable one.
        """
        n_components, feature_dims, rank = 3, 6, 2
        x = np.random.RandomState(31).normal(
            size=(8, feature_dims)
        ).astype("float32")
        layer = GMMLayer(
            n_components=n_components,
            covariance_type="low_rank",
            covariance_rank=rank,
            mean_initializer="glorot_normal",
        )
        layer(x)  # build

        # Pin to a known, clearly non-initializer value so "changed" is unambiguous.
        pinned = np.full((n_components, feature_dims, rank), 7.0, dtype="float32")
        layer.covariance_factors.assign(pinned)

        layer.reset_parameters()
        after = np.asarray(ops.convert_to_numpy(layer.covariance_factors))

        assert after.shape == (n_components, feature_dims, rank)
        assert not np.allclose(after, pinned), (
            "reset_parameters() left covariance_factors untouched; a reset low-rank "
            "layer would silently retain its old covariance structure (D-005)"
        )
        assert np.abs(after).max() > 0.0, (
            "reset_parameters() zeroed covariance_factors. Zero U is a DEAD weight -- "
            "the Woodbury correction is quadratic in U about the origin, so dL/dU "
            "vanishes identically. Re-run factor_initializer instead (D-005)."
        )
        assert np.all(np.isfinite(after))

    def test_reset_is_inert_for_diagonal(self) -> None:
        """The D-005 fix must not disturb the diagonal path: no weight, no crash."""
        x = np.random.RandomState(32).normal(size=(8, 6)).astype("float32")
        layer = GMMLayer(n_components=3, mean_initializer="glorot_normal")
        layer(x)
        assert layer.covariance_factors is None

        layer.reset_parameters()  # must not raise

        np.testing.assert_allclose(
            ops.convert_to_numpy(layer.log_variances),
            np.zeros_like(ops.convert_to_numpy(layer.log_variances)),
            atol=1e-7,
        )

    def test_gradients_still_flow_after_reset(self) -> None:
        """D-005's ACTUAL claim: the post-reset weight is still TRAINABLE.

        ``test_reset_reinitializes_covariance_factors`` asserts the values changed and
        are non-zero. Neither is what D-005 is about. The rejected fix (zeroing ``U``)
        also "changes" the values, and "non-zero" is a proxy for trainability, not a
        measurement of it -- a hypothetical reset that produced some other degenerate
        ``U`` would pass both assertions and still be dead. This test measures the
        property directly: run a fresh ``GradientTape`` AFTER the reset and require a
        non-``None``, non-zero gradient. A zeroing reset fails here (``dL/dU`` vanishes
        identically at ``U = 0``, since the Woodbury correction is quadratic about the
        origin), which is exactly the failure D-005 exists to prevent.
        """
        import tensorflow as tf

        n_components, feature_dims, rank, batch = 3, 6, 2, 12
        rng = np.random.RandomState(41)
        x = rng.normal(size=(batch, feature_dims)).astype("float32")
        target = rng.uniform(size=(batch, n_components)).astype("float32")

        layer = GMMLayer(
            n_components=n_components,
            covariance_type="low_rank",
            covariance_rank=rank,
            mean_initializer="glorot_normal",
        )
        layer(x)  # build
        layer.reset_parameters()

        with tf.GradientTape() as tape:
            out = layer(x, training=True)
            loss = tf.reduce_mean(tf.square(out - target))
            if layer.losses:
                loss = loss + tf.add_n(layer.losses)
        grad = tape.gradient(loss, layer.covariance_factors)

        assert grad is not None, (
            "covariance_factors has a None gradient after reset_parameters() -- the "
            "reset disconnected the weight from the loss (D-005)"
        )
        grad_np = np.asarray(ops.convert_to_numpy(grad), dtype=np.float64)
        assert np.all(np.isfinite(grad_np))
        assert np.abs(grad_np).max() > 0.0, (
            "covariance_factors gradient is identically zero after reset_parameters(). "
            "The reset produced a DEAD U (this is precisely what zeroing would do). "
            "reset_parameters() must re-run factor_initializer (D-005)."
        )


# ------------------------------------------------------- low-rank coverage gaps
# Added at REFLECT: the iteration's suite was green while under-testing these
# interactions. Each test below targets one gap named by the iteration-1 review.

class TestGMMLayerLowRankInteractions:
    """Interactions the low-rank path was never exercised under."""

    def test_low_rank_multi_axis_cluster_axis(self) -> None:
        """Gap 2: ``low_rank`` under a MULTI-AXIS ``cluster_axis``, end to end.

        These two features had zero joint coverage, and the base-class extraction moved
        exactly the seam that couples them: the low-rank branch of ``build()`` sizes
        ``U_k`` from ``feature_dims``, which is derived from the normalized
        ``cluster_axis``, while ``compute_output_shape`` re-derives its own axes from the
        PRE-build ``_cluster_axis_arg``. If those two derivations ever disagree, the
        single-axis tests cannot see it.

        This test covers the parts of that seam that are currently correct: the
        pre-build and post-build shape contract, ``feature_dims``, and the sizing of
        ``U_k``. The VALUE-level contract is broken by a pre-existing defect in the
        shared multi-axis output reshape and is pinned separately by
        ``test_multi_axis_output_layout_is_scrambled`` below.
        """
        n_components, rank = 4, 2
        shape = (4, 5, 7, 3)                # cluster over axes 1 and 2 -> D = 35
        feature_dims = shape[1] * shape[2]
        x = np.random.RandomState(51).normal(size=shape).astype("float32")

        layer = GMMLayer(
            n_components=n_components,
            covariance_type="low_rank",
            covariance_rank=rank,
            cluster_axis=[1, 2],
            mean_initializer="glorot_normal",
            log_variance_initializer="glorot_normal",
        )

        # Shape contract must hold BEFORE build (functional-API tracing path)...
        expected_shape = (shape[0], n_components, shape[3])
        assert layer.compute_output_shape(shape) == expected_shape

        y = layer(x)
        assert tuple(y.shape) == expected_shape
        # ...and again after build, where cluster_axis has been mutated in place.
        assert layer.compute_output_shape(shape) == expected_shape

        assert layer.feature_dims == feature_dims
        assert tuple(layer.covariance_factors.shape) == (
            n_components, feature_dims, rank
        ), "U_k was sized from the wrong feature_dims under a multi-axis cluster_axis"

        y_np = np.asarray(ops.convert_to_numpy(y), dtype=np.float64)
        assert np.all(np.isfinite(y_np))
        assert np.all(y_np >= 0.0)

    # DECISION plan-2026-07-20T141712-e03557c8/D-010: characterize and PIN, do not fix.
    # See src/dl_techniques/layers/mixtures/base.py::_reshape_output for the mechanism and
    # decisions.md D-010 for the full record. strict=True so a repair flips this to a
    # hard FAILURE demanding attention, rather than letting the defect silently persist.
    @pytest.mark.xfail(
        strict=True,
        reason=(
            "PRE-EXISTING DEFECT, not introduced by the low-rank work and not specific "
            "to it. Under a multi-axis cluster_axis, _reshape_output (base.py, lifted "
            "byte-identically from gmm.py/kmeans.py at bf953d6c) builds the correct "
            "output SHAPE but applies a bare reshape to a buffer laid out "
            "(batch, non_feature, K). The declared axis order is (batch, K, "
            "non_feature), so the two trailing axes are swapped -- and because it is a "
            "reshape rather than a transpose, the values are outright scrambled "
            "whenever K != non_feature (they merely transpose when K == non_feature, "
            "which is why every existing multi-axis test and the iteration-1 reviewer's "
            "spot-check missed it: all used K == non_feature). Confirmed present at "
            "baseline bf953d6c and shared with KMeansLayer. Fixing it is a production "
            "behavior change and is out of scope for this step; xfail(strict) so this "
            "flips to a FAILURE the moment someone repairs it."
        ),
    )
    def test_multi_axis_output_layout_is_scrambled(self) -> None:
        """Gap 2 (value contract): multi-axis responsibilities must be correctly placed.

        The oracle is an INDEPENDENT reimplementation of the flatten/unflatten, not a
        self-consistency check: the same data is transposed and flattened by hand and
        fed to a plain 2-D low-rank layer carrying byte-identical weights.

        ``n_components=4`` against a trailing non-feature dim of 3 is deliberate. With
        ``K == non_feature`` the defect degenerates into a pure transposition and can
        hide behind symmetric test shapes; with ``K != non_feature`` the reshape mixes
        the axes outright, which the softmax check below exposes directly --
        responsibilities stop summing to 1 along ANY axis.
        """
        n_components, rank = 4, 2
        shape = (4, 5, 7, 3)
        feature_dims = shape[1] * shape[2]
        x = np.random.RandomState(51).normal(size=shape).astype("float32")

        common: Dict[str, Any] = {
            "n_components": n_components,
            "covariance_type": "low_rank",
            "covariance_rank": rank,
            "mean_initializer": "glorot_normal",
            "log_variance_initializer": "glorot_normal",
        }
        layer = GMMLayer(cluster_axis=[1, 2], **common)
        y = np.asarray(ops.convert_to_numpy(layer(x)), dtype=np.float64)

        # The cheapest independent symptom: axis 1 is DECLARED to be the component
        # axis, so responsibilities must sum to 1 along it.
        np.testing.assert_allclose(
            y.sum(axis=1), 1.0, rtol=1e-5, atol=1e-5,
            err_msg=(
                "multi-axis responsibilities do not sum to 1 along the declared "
                "component axis -- the output buffer is mis-laid-out"
            ),
        )

        flat = GMMLayer(**common)
        x_flat = np.transpose(x, (0, 3, 1, 2)).reshape(-1, feature_dims)
        flat(x_flat)  # build
        for name in _SHARED_PARAM_NAMES + ("covariance_factors",):
            getattr(flat, name).assign(ops.convert_to_numpy(getattr(layer, name)))

        reference = np.asarray(ops.convert_to_numpy(flat(x_flat)), dtype=np.float64)
        # (batch * non_feature, K) -> (batch, non_feature, K) -> (batch, K, non_feature)
        reference = np.transpose(
            reference.reshape(shape[0], shape[3], n_components), (0, 2, 1)
        )

        np.testing.assert_allclose(
            y, reference, rtol=1e-6, atol=1e-6,
            err_msg=(
                "multi-axis low-rank responsibilities disagree with a hand-flattened "
                "2-D reference: the cluster-axis transpose/regroup contract is wrong"
            ),
        )

    def test_low_rank_mixed_float16_forward(self) -> None:
        """Gap 3: ``low_rank`` under ``mixed_float16``.

        ``test_mixed_float16_forward`` covers the diagonal path only, while the plan
        flags ``cholesky`` / ``solve_triangular`` as the MORE fp16-fragile ops. The
        layer's defense is that it casts inputs to ``variable_dtype`` and runs the whole
        density in float32, so the only lossy step should be the final cast of the
        responsibilities to float16.

        That is a testable claim with a principled bound rather than a tuned one: if the
        defense holds, the fp16 output must agree with an identically-weighted float32
        layer to within float16 quantization of values in ``[0, 1]`` (1 ulp at 1.0 is
        ``2**-10`` = 9.77e-4). If any part of the Woodbury/Cholesky path silently ran in
        half precision, the error would be orders of magnitude larger than the cast
        bound and this fails. Measured deviation at the time of writing: 3.4e-04.
        """
        float16_ulp_at_one = 2.0 ** -10  # 9.766e-04

        x = np.random.RandomState(61).normal(size=(8, 16)).astype("float32")
        layer_kwargs: Dict[str, Any] = {
            "n_components": 4,
            "covariance_type": "low_rank",
            "covariance_rank": 3,
            "mean_initializer": "glorot_normal",
            "log_variance_initializer": "glorot_normal",
        }

        original_policy = keras.mixed_precision.global_policy()
        try:
            keras.mixed_precision.set_global_policy("mixed_float16")
            half = GMMLayer(**layer_kwargs)
            y_half = half(x)
            assert keras.backend.standardize_dtype(y_half.dtype) == "float16"
            # Weights must stay float32 (autocast=False) -- including the new one.
            for name in _SHARED_PARAM_NAMES + ("covariance_factors",):
                assert keras.backend.standardize_dtype(
                    getattr(half, name).dtype
                ) == "float32", f"{name} was created in the compute dtype, not float32"
            y_half_np = np.asarray(ops.convert_to_numpy(y_half), dtype=np.float64)
            assert np.all(np.isfinite(y_half_np)), (
                "low-rank forward produced NaN/Inf under mixed_float16"
            )
        finally:
            keras.mixed_precision.set_global_policy(original_policy)

        full = GMMLayer(**layer_kwargs)
        full(x)  # build under the restored float32 policy
        for name in _SHARED_PARAM_NAMES + ("covariance_factors",):
            getattr(full, name).assign(ops.convert_to_numpy(getattr(half, name)))
        y_full_np = np.asarray(ops.convert_to_numpy(full(x)), dtype=np.float64)

        max_deviation = float(np.abs(y_half_np - y_full_np).max())
        assert max_deviation <= float16_ulp_at_one, (
            f"mixed_float16 low-rank output deviates from the float32 reference by "
            f"{max_deviation:.3e}, above one float16 ulp at 1.0 ({float16_ulp_at_one:.3e}). "
            f"That is larger than an output-cast can explain -- suspect the Cholesky / "
            f"solve_triangular path running in half precision. Do NOT raise this bound."
        )

    def test_covariance_rank_at_or_above_feature_dims(self, caplog) -> None:
        """Gap 5: ``covariance_rank >= feature_dims`` is legal-but-uneconomical.

        ``build()`` warns and proceeds. Both halves are asserted: the warning must
        actually reach the log (it is the user's only signal that they are paying for a
        parameterization with no compression benefit), and the math must stay correct --
        an over-ranked ``U`` makes the ``(R, R)`` capacitance matrix rank-deficient in
        its ``U^T diag(1/d) U`` term, so the ``I_R +`` that guarantees positive
        definiteness is doing all the work. If that guarantee were wrong, ``cholesky``
        would return NaN here rather than raise, so finiteness is the real assertion.
        """
        import logging

        n_components, feature_dims, rank = 3, 4, 8
        x = np.random.RandomState(71).normal(
            size=(6, feature_dims)
        ).astype("float32")

        with caplog.at_level(logging.WARNING, logger="dl"):
            layer = GMMLayer(
                n_components=n_components,
                covariance_type="low_rank",
                covariance_rank=rank,
                mean_initializer="glorot_normal",
                log_variance_initializer="glorot_normal",
            )
            y = layer(x)

        assert any(
            ">= feature_dims" in record.message
            for record in caplog.records
            if record.levelno == logging.WARNING
        ), (
            f"covariance_rank={rank} >= feature_dims={feature_dims} must emit a "
            f"logger.warning; the user has no other signal that the low-rank "
            f"parameterization is buying nothing here"
        )

        assert tuple(layer.covariance_factors.shape) == (
            n_components, feature_dims, rank
        )
        assert tuple(y.shape) == (6, n_components)
        y_np = np.asarray(ops.convert_to_numpy(y), dtype=np.float64)
        assert np.all(np.isfinite(y_np)), (
            "over-ranked low-rank covariance produced NaN/Inf -- the I_R + PSD "
            "positive-definiteness guarantee (invariant I6) does not hold"
        )
        np.testing.assert_allclose(y_np.sum(axis=-1), 1.0, rtol=1e-6, atol=1e-6)

    def test_trained_weights_survive_save_load(self) -> None:
        """Gap 6: the guide-§8.2 round-trip on TRAINED weights, not initializer output.

        ``test_model_save_load_low_rank`` saves a freshly-built layer. Fresh weights are
        the one case a restore bug can hide in: a loader that silently re-runs the
        initializers instead of restoring the checkpoint, or that mixes up two weights of
        compatible shape, can still reproduce predictions when every weight is close to
        its initialized value. Training first breaks that symmetry -- the four weights
        are then mutually distinguishable and far from any initializer's output, so a
        mis-restore shows up as a numerical difference.
        """
        n_components, feature_dims, rank = 3, 8, 2
        rng = np.random.RandomState(81)
        x = rng.normal(size=(24, feature_dims)).astype("float32")
        target = rng.uniform(size=(24, n_components)).astype("float32")

        inp = keras.Input(shape=(feature_dims,))
        gmm = GMMLayer(
            n_components=n_components,
            covariance_type="low_rank",
            covariance_rank=rank,
            factor_initializer="he_normal",
            mean_initializer="glorot_normal",
            name="gmm",
        )
        model = keras.Model(inp, gmm(inp))

        before = {
            name: np.array(ops.convert_to_numpy(getattr(gmm, name)))
            for name in _SHARED_PARAM_NAMES + ("covariance_factors",)
        }

        model.compile(optimizer=keras.optimizers.Adam(0.05), loss="mse")
        model.fit(x, target, epochs=2, batch_size=8, verbose=0)

        # The premise of this test: every weight must actually have MOVED, or it
        # degenerates into the fresh-weights round-trip it is meant to strengthen.
        for name, old in before.items():
            new = np.asarray(ops.convert_to_numpy(getattr(gmm, name)))
            assert not np.allclose(new, old, rtol=1e-4, atol=1e-4), (
                f"'{name}' did not move during training, so this round-trip is no "
                f"stronger than the fresh-weights one"
            )

        y0 = model(x)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "gmm_trained.keras")
            model.save(path)
            loaded = keras.models.load_model(path)

        np.testing.assert_allclose(
            ops.convert_to_numpy(y0),
            ops.convert_to_numpy(loaded(x)),
            rtol=1e-6, atol=1e-6,
            err_msg=(
                "trained low-rank weights did not survive a .keras round-trip "
                "(the freshly-built round-trip cannot catch this)"
            ),
        )
