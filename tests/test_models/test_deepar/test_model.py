"""
Test suite for the DeepAR probabilistic forecaster + its training wrapper.

These tests exercise the iteration-4 deliverables that join DeepAR to the
unified `Forecast`/`ForecastMixin` contract:

1. `DeepAR.get_config` round-trip POST-mixin -- a regression guard that mixing
   in the stateless `ForecastMixin` did not break serialization.
2. `DeepAR._forecast` shapes -- the Monte-Carlo sample -> mean point +
   empirical-percentile quantiles + populated `Forecast.samples` contract.
3. `DeepARTrainingWrapper.get_config` round-trip -- the wrapper serializes /
   re-instantiates its base DeepAR (the D-001 add_loss wrapper).
4. `DeepARTrainingWrapper` add_loss presence -- a training-mode forward
   registers the NLL via `add_loss` (so `compile(loss=None)` trains).

All dims are deliberately tiny so the eager autoregressive sampling loop in
`_forecast` stays CPU-fast. numpy is seeded for determinism. Layer/loss numerics
are covered elsewhere and are NOT re-tested here.
"""

import os
import keras
import pytest
import numpy as np

from dl_techniques.models.time_series.deepar.model import DeepAR, create_deepar
from dl_techniques.models.time_series.forecast import Forecast, ForecastMixin
from train.time_series.deepar.train_deepar import DeepARTrainingWrapper
from dl_techniques.utils.logger import logger


# Small, CPU-safe problem geometry shared across tests.
B = 2            # batch
L = 12           # conditioning / input length
H = 4            # prediction horizon
T = L + H        # teacher-forced window
D = 1            # target_dim
C = 4            # covariate_dim
S = 4            # num_samples (small -> fast eager loop)


class TestDeepAR:
    """DeepAR base-model tests (post-ForecastMixin)."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(0)
        keras.utils.set_random_seed(0)

    @pytest.fixture
    def model_config(self):
        return dict(
            num_layers=1,
            hidden_dim=8,
            likelihood="gaussian",
            target_dim=D,
            num_samples=S,
        )

    @pytest.fixture
    def built_model(self, model_config):
        """A DeepAR built via a training-mode dummy dict forward."""
        model = DeepAR(**model_config)
        dummy = {
            "target": np.zeros((1, T, D), dtype="float32"),
            "covariates": np.zeros((1, T, C), dtype="float32"),
        }
        model(dummy, training=False)  # build via training-mode forward
        return model

    # -- 1: get_config round-trip post-mixin -------------------------------
    def test_get_config_round_trip_post_mixin(self, model_config):
        """from_config rebuilds; the ForecastMixin must not break get_config."""
        model = DeepAR(**model_config)
        assert isinstance(model, ForecastMixin)  # mixin is on the bases

        config = model.get_config()
        rebuilt = DeepAR.from_config(config)

        for key, val in model_config.items():
            assert getattr(rebuilt, key) == val, f"{key} not preserved"
        assert rebuilt.num_layers == model_config["num_layers"]
        assert rebuilt.hidden_dim == model_config["hidden_dim"]
        assert rebuilt.likelihood == model_config["likelihood"]
        logger.info("DeepAR get_config round-trip (post-mixin) test passed.")

    # -- 2: _forecast shapes ----------------------------------------------
    def test_forecast_shapes(self, built_model):
        """_forecast returns point [B,H,D], quantiles [B,H,D,Q], non-None samples."""
        pred_dict = {
            "conditioning_target": np.random.randn(B, L, D).astype("float32"),
            "full_covariates": np.random.randn(B, T, C).astype("float32"),
        }
        fc = built_model._forecast(pred_dict)

        assert isinstance(fc, Forecast)
        q = len(fc.quantile_levels)  # default [0.1, 0.5, 0.9] -> 3
        assert q == 3
        assert fc.point.shape == (B, H, D), f"point shape {fc.point.shape}"
        assert fc.quantiles.shape == (B, H, D, q), (
            f"quantiles shape {fc.quantiles.shape}"
        )
        assert fc.samples is not None
        assert fc.samples.shape == (S, B, H, D), f"samples shape {fc.samples.shape}"
        assert np.all(np.isfinite(fc.point))
        logger.info("DeepAR _forecast shapes test passed.")

    # -- 5: .keras save/load round-trip for DeepAR itself ------------------
    def test_keras_save_load_round_trip(self, built_model, tmp_path):
        """DeepAR saves to .keras and reloads with numeric-equal training output.

        The model is BUILT (via the built_model fixture's dummy forward) before
        save so the never-called-subclass save warning does not fire. We compare
        the deterministic training-mode `mu` (NOT the stochastic sampling path).
        """
        train_in = {
            "target": np.random.randn(B, T, D).astype("float32"),
            "covariates": np.random.randn(B, T, C).astype("float32"),
        }
        out_before = built_model(train_in, training=False)["mu"]

        path = os.path.join(str(tmp_path), "deepar.keras")
        built_model.save(path)
        reloaded = keras.models.load_model(path)

        assert isinstance(reloaded, DeepAR)
        out_after = reloaded(train_in, training=False)["mu"]
        np.testing.assert_allclose(
            np.asarray(out_before), np.asarray(out_after), atol=1e-6
        )
        logger.info("DeepAR .keras save/load round-trip test passed.")

    # -- 6: weight-count survives a save/reload rebuild -------------------
    def test_weight_count_after_rebuild(self, built_model, tmp_path):
        """Weights are tracked (non-empty) and preserved across save/reload.

        Guards the E3/A2 empirical unknown: the plain-Python-list `lstm_layers`
        IS tracked by Keras 3, so the reloaded model must have the same weight
        count as the original (LSTM kernels included).
        """
        n_before = len(built_model.weights)
        assert n_before > 0, "DeepAR has zero tracked weights after build"

        path = os.path.join(str(tmp_path), "deepar_wc.keras")
        built_model.save(path)
        reloaded = keras.models.load_model(path)

        assert len(reloaded.weights) == n_before, (
            f"weight count changed across rebuild: {n_before} -> "
            f"{len(reloaded.weights)}"
        )
        # LSTM weights must be present (list-tracking confirmation).
        lstm_w = [w for w in reloaded.weights if "lstm" in w.path.lower()]
        assert len(lstm_w) > 0, "LSTM weights not tracked after rebuild"
        logger.info("DeepAR weight-count-after-rebuild test passed.")

    # -- 7: create_deepar factory -----------------------------------------
    def test_create_deepar_factory(self):
        """create_deepar returns a BUILT DeepAR with non-empty weights."""
        model = create_deepar(
            num_layers=1, hidden_dim=8, likelihood="gaussian",
            target_dim=D, num_samples=S, covariate_dim=C,
        )
        assert isinstance(model, DeepAR)
        assert len(model.weights) > 0, "factory model not built"
        assert model.hidden_dim == 8
        logger.info("create_deepar factory test passed.")

    # -- 8: ctor validation -----------------------------------------------
    @pytest.mark.parametrize(
        "bad_kwargs",
        [
            {"num_layers": 0},
            {"hidden_dim": 0},
            {"target_dim": 0},
            {"num_samples": 0},
            {"likelihood": "not_a_likelihood"},
        ],
    )
    def test_ctor_validation_raises(self, bad_kwargs):
        """Non-positive geometry / unknown likelihood raise ValueError."""
        kwargs = dict(
            num_layers=1, hidden_dim=8, likelihood="gaussian",
            target_dim=D, num_samples=S,
        )
        kwargs.update(bad_kwargs)
        with pytest.raises(ValueError):
            DeepAR(**kwargs)
        logger.info(f"DeepAR ctor validation rejected {bad_kwargs}.")


class TestDeepARTrainingWrapper:
    """DeepARTrainingWrapper tests (the D-001 add_loss wrapper)."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(0)
        keras.utils.set_random_seed(0)

    @pytest.fixture
    def dummy_train_inputs(self):
        return {
            "target": np.random.randn(B, T, D).astype("float32"),
            "covariates": np.random.randn(B, T, C).astype("float32"),
        }

    @pytest.fixture
    def built_wrapper(self, dummy_train_inputs):
        base = DeepAR(
            num_layers=1, hidden_dim=8, likelihood="gaussian",
            target_dim=D, num_samples=S,
        )
        wrapper = DeepARTrainingWrapper(base)
        wrapper(dummy_train_inputs, training=True)  # build the wrapper
        return wrapper

    # -- 3: wrapper get_config round-trip ---------------------------------
    def test_wrapper_get_config_round_trip(self, built_wrapper):
        """from_config(get_config()) rebuilds; rebuilt .base is a DeepAR."""
        config = built_wrapper.get_config()
        rebuilt = DeepARTrainingWrapper.from_config(config)

        assert isinstance(rebuilt, DeepARTrainingWrapper)
        assert isinstance(rebuilt.base, DeepAR)
        assert rebuilt.base.likelihood == built_wrapper.base.likelihood
        assert rebuilt.base.hidden_dim == built_wrapper.base.hidden_dim
        logger.info("DeepARTrainingWrapper get_config round-trip test passed.")

    # -- 4: wrapper add_loss present --------------------------------------
    def test_wrapper_registers_loss(self, dummy_train_inputs):
        """A training-mode call registers the NLL via add_loss (len(losses) > 0)."""
        base = DeepAR(
            num_layers=1, hidden_dim=8, likelihood="gaussian",
            target_dim=D, num_samples=S,
        )
        wrapper = DeepARTrainingWrapper(base)
        wrapper(dummy_train_inputs, training=True)

        assert len(wrapper.losses) > 0, "wrapper.call did not add_loss the NLL"
        assert all(np.isfinite(float(l)) for l in wrapper.losses)
        logger.info("DeepARTrainingWrapper add_loss presence test passed.")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ---------------------------------------------------------------------


class TestDeepARScaleSemantics:
    """The scale must be a first-moment scale, and must not see the future.

    Both assertions are derived from Salinas et al. §3.2 (mu = nu*mu~,
    sigma = nu*softplus(sigma~)), NOT read off the implementation.
    """

    CFG = dict(num_layers=1, hidden_dim=8, likelihood='gaussian', target_dim=1)

    @staticmethod
    def _inputs(target, cov):
        return {'target': target, 'covariates': cov}

    def test_sigma_scales_linearly_with_the_target(self):
        """Scaling the target by k must scale BOTH mu and sigma by k.

        RED-proof: sigma was de-normalized with sqrt(scale) while mu used
        scale, so sigma grew as sqrt(k) and every predictive interval was
        mis-calibrated by a scale-dependent factor. Asserting sigma > 0, or
        that it is finite, would be VACUOUS -- both held.

        Homogeneity is the right probe because it is a property of the
        parameterization and needs no reference implementation to check.
        """
        keras.utils.set_random_seed(0)
        rng = np.random.default_rng(0)
        T = 12
        base = np.abs(rng.normal(size=(4, T, 1))).astype("float32") + 1.0
        cov = rng.normal(size=(4, T, 3)).astype("float32")

        model = DeepAR(**self.CFG)
        # A precomputed scale is supplied so this test isolates the
        # de-normalization arithmetic from the scale-estimation question.
        k = 10.0
        s1 = np.mean(base, axis=1, keepdims=True).astype("float32")
        s2 = (s1 * k).astype("float32")

        o1 = model({**self._inputs(base, cov), 'scale': s1}, training=False)
        o2 = model({**self._inputs(base * k, cov), 'scale': s2}, training=False)

        mu1 = keras.ops.convert_to_numpy(o1['mu'])
        mu2 = keras.ops.convert_to_numpy(o2['mu'])
        sg1 = keras.ops.convert_to_numpy(o1['sigma'])
        sg2 = keras.ops.convert_to_numpy(o2['sigma'])

        np.testing.assert_allclose(mu2, mu1 * k, rtol=1e-4,
                                   err_msg="mu is not scale-linear")
        np.testing.assert_allclose(
            sg2, sg1 * k, rtol=1e-4,
            err_msg=("sigma is not scale-linear: it is a first-moment-scale "
                     "quantity, not a variance, so it must scale by nu and not "
                     "by sqrt(nu)"))

    def test_scale_ignores_the_prediction_range(self):
        """With conditioning_length set, mutating the TAIL must not move nu.

        RED-proof: compute_scale was called without conditioning_length, so nu
        was mean(z_1..z_T) over the whole window and every teacher-forced step
        was divided by a function of its own future.
        """
        rng = np.random.default_rng(1)
        T, C = 12, 6
        target = np.abs(rng.normal(size=(2, T, 1))).astype("float32") + 1.0
        mutated = target.copy()
        mutated[:, C:, :] += 100.0  # only the prediction range changes

        cov = rng.normal(size=(2, T, 3)).astype("float32")

        keras.utils.set_random_seed(0)
        model = DeepAR(**self.CFG, conditioning_length=C)

        # Go through call(), not compute_scale directly: compute_scale has
        # ALWAYS accepted conditioning_length, so probing it in isolation would
        # pass with or without the fix. What was broken is the WIRING -- no
        # caller ever passed the argument.
        o_base = model(self._inputs(target, cov), training=False)
        o_mut = model(self._inputs(mutated, cov), training=False)

        mu_base = keras.ops.convert_to_numpy(o_base['mu'])[:, :C, :]
        mu_mut = keras.ops.convert_to_numpy(o_mut['mu'])[:, :C, :]

        np.testing.assert_allclose(
            mu_base, mu_mut, rtol=1e-5, atol=1e-6,
            err_msg=("mu over the CONDITIONING range moved when only the "
                     "PREDICTION range was mutated -- the scale is leaking the "
                     "future into every teacher-forced step"))

        # Control: the perturbation is real. A model with no conditioning split
        # DOES move, so the agreement above is attributable to the split.
        keras.utils.set_random_seed(0)
        leaky = DeepAR(**self.CFG)
        l_base = keras.ops.convert_to_numpy(
            leaky(self._inputs(target, cov), training=False)['mu'])[:, :C, :]
        l_mut = keras.ops.convert_to_numpy(
            leaky(self._inputs(mutated, cov), training=False)['mu'])[:, :C, :]
        assert not np.allclose(l_base, l_mut, rtol=1e-5, atol=1e-6), (
            "even without a conditioning split the tail perturbation changed "
            "nothing; the isolation assertion above would be vacuous")
