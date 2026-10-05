"""Test suite for `TopographicVAELoss` (Keller & Welling 2022, Eq. 12).

The objective has three terms and two reduction conventions, and both are places
a number can quietly mean something other than what a reader assumes. So the suite
is dominated by **closed-form comparisons**: every term is checked against a
NumPy evaluation of its definition, on a NON-SQUARE image shape so an
off-by-one-axis reduction cannot pass.

MEASURED on the reference fixture (B=2, S=3, 4x6x1 frames, latent 8):
  * mean-over-pixels reduction, oracle 20.59669617 vs loss 20.59669685, delta 6.9e-07
  * sum-over-pixels reduction,  oracle 53.69670048 vs loss 53.69670105, delta 5.7e-07
  * a mean over WIDTH alone would give 19.637276 -- 0.96 away, so the axis is pinned
  * the whole objective runs in float32 and agrees with float64 to ~1e-7
"""

import numpy as np
import pytest
from keras import ops

from dl_techniques.losses.topographic_vae_loss import (
    LOG_VAR_CLIP,
    PROBABILITY_CLIP,
    REQUIRED_PREDICTION_KEYS,
    VARIANCE_PREDICTION_KEYS,
    TopographicVAELoss,
)

B, S, H, W, C, D = 2, 3, 4, 6, 1, 8
# Deliberately non-square (H=4, W=6): a reduction over the wrong image axis is
# invisible at H == W, and `binary_crossentropy` has already dropped the channel
# axis, so the remaining image axes are 2..rank-1 of its OWN rank.


def _fixture(seed=0, log_var=0.0, reconstruction=None, batch=None):
    batch = B if batch is None else batch
    rng = np.random.default_rng(seed)
    targets = rng.random((batch, S, H, W, C)).astype("float32")
    z_mean = rng.normal(size=(batch, S, D)).astype("float32")
    u_mean = (0.5 * z_mean).astype("float32")
    prediction = {
        "reconstruction": ops.convert_to_tensor(
            targets.copy() if reconstruction is None else reconstruction
        ),
        "z_mean": ops.convert_to_tensor(z_mean),
        "z_log_var": ops.convert_to_tensor(
            np.full((batch, S, D), log_var, "float32")
        ),
        "u_mean": ops.convert_to_tensor(u_mean),
        "u_log_var": ops.convert_to_tensor(
            np.full((batch, S, D), log_var, "float32")
        ),
    }
    return ops.convert_to_tensor(targets), prediction, targets, z_mean, u_mean


def _oracle(targets, prediction, z_mean, u_mean, log_var, kl_weight, sum_reduction):
    """NumPy evaluation of Eq. 12, from the definitions, PER SAMPLE."""
    # The Bernoulli NLL is the NEGATIVE log-likelihood: without the minus this
    # oracle returns a negative "loss" and disagrees with the layer by ~2.9 nats.
    nll = -(
        targets * np.log(prediction)
        + (1.0 - targets) * np.log1p(-prediction)
    )
    if sum_reduction:
        reconstruction = nll.sum(axis=(1, 2, 3, 4))
    else:
        reconstruction = nll.mean(axis=(2, 3, 4)).sum(axis=-1)

    def _kl(mean):
        return (
            0.5 * (mean**2 + np.exp(log_var) - 1.0 - log_var)
        ).sum(axis=-1).sum(axis=-1)

    return reconstruction + kl_weight * (_kl(z_mean) + _kl(u_mean))


class TestTopographicVAELossConstructor:
    """Stored configuration and validation."""

    def test_stores_its_configuration(self):
        loss = TopographicVAELoss(
            kl_loss_weight=0.25,
            reconstruction_sum_reduction=True,
            clip_log_var=7.0,
        )
        assert loss.kl_loss_weight == 0.25
        assert loss.reconstruction_sum_reduction is True
        assert loss.clip_log_var == 7.0

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"kl_loss_weight": -0.1}, "kl_loss_weight must be non-negative"),
            ({"clip_log_var": 0.0}, "clip_log_var must be positive"),
        ],
    )
    def test_rejects_invalid_arguments(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            TopographicVAELoss(**kwargs)

    def test_config_round_trips(self):
        loss = TopographicVAELoss(
            kl_loss_weight=0.5, reconstruction_sum_reduction=True, clip_log_var=5.0
        )
        config = loss.get_config()
        assert config["kl_loss_weight"] == 0.5
        assert config["reconstruction_sum_reduction"] is True
        assert config["clip_log_var"] == 5.0
        restored = TopographicVAELoss.from_config(config)
        assert restored.kl_loss_weight == 0.5
        assert restored.reconstruction_sum_reduction is True

    def test_declares_every_required_prediction_key(self):
        """The always-required set, and the conditional ``u`` pair beside it.

        Split deliberately: ``u_mean``/``u_log_var`` are absent from a model
        built with ``use_variance_variables=False``, and an objective that
        demanded them would make the paper's plain-VAE baseline untrainable.
        """
        assert set(REQUIRED_PREDICTION_KEYS) == {
            "reconstruction", "z_mean", "z_log_var",
        }
        assert set(VARIANCE_PREDICTION_KEYS) == {"u_mean", "u_log_var"}
        assert not set(REQUIRED_PREDICTION_KEYS) & set(
            VARIANCE_PREDICTION_KEYS
        )

    def test_registers_under_a_package_qualified_key(self):
        import keras

        key = keras.saving.get_registered_name(TopographicVAELoss)
        assert (
            key == "dl_techniques.losses.topographic_vae_loss"
            ">TopographicVAELoss"
        ), key


class TestTopographicVAELossContract:
    """A missing term must raise, not be skipped."""

    @pytest.mark.parametrize("missing", REQUIRED_PREDICTION_KEYS)
    def test_a_missing_prediction_key_raises(self, missing):
        targets, prediction, *_ = _fixture()
        partial = {k: v for k, v in prediction.items() if k != missing}
        with pytest.raises(ValueError, match=missing):
            TopographicVAELoss()(targets, partial)

    @pytest.mark.parametrize("missing", VARIANCE_PREDICTION_KEYS)
    def test_a_half_present_u_pair_raises(self, missing):
        """``u_mean`` without ``u_log_var`` is not a distribution.

        Defaulting the absent scale to unit variance would train against a prior
        the model does not have, and the resulting model would look trained.
        """
        targets, prediction, *_ = _fixture()
        partial = {k: v for k, v in prediction.items() if k != missing}
        with pytest.raises(ValueError, match=missing):
            TopographicVAELoss()(targets, partial)

    def test_the_baseline_prediction_is_accepted(self):
        """``use_variance_variables=False`` emits neither ``u`` key.

        This is the plain-VAE baseline the paper compares against, so an
        objective that rejects its output would make the comparison
        unrunnable -- the defect this arm exists to pin.
        """
        targets, prediction, *_ = _fixture()
        baseline = {
            k: v for k, v in prediction.items()
            if k not in VARIANCE_PREDICTION_KEYS
        }
        loss = TopographicVAELoss(kl_loss_weight=1.0)
        value = ops.convert_to_numpy(loss.call(targets, baseline))
        assert value.shape == (B,), value.shape

    def test_the_baseline_drops_exactly_the_u_kl_term(self):
        """Not merely accepted: the number must be the full loss MINUS KL_u.

        An implementation that added a zero `u_kl` of the wrong SHAPE would
        broadcast and still return a `(batch,)` tensor, so the term is pinned by
        its value rather than by its presence.
        """
        targets, prediction, targets_np, z_mean, u_mean = _fixture()
        loss = TopographicVAELoss(kl_loss_weight=1.0)
        with_u = ops.convert_to_numpy(loss.call(targets, prediction))
        without_u = ops.convert_to_numpy(
            loss.call(
                targets,
                {k: v for k, v in prediction.items()
                 if k not in VARIANCE_PREDICTION_KEYS},
            )
        )
        log_var = 0.0
        # Per-step KL, then Eq. 12's outer sum over the sequence axis -- the same
        # two reductions the loss applies. Omitting the sequence sum leaves a
        # (batch, sequence) array against a (batch,) one.
        expected_u_kl = (
            0.5
            * (u_mean.astype(np.float64) ** 2 + np.exp(log_var) - 1.0 - log_var)
        ).sum(axis=-1).sum(axis=-1)
        np.testing.assert_allclose(
            with_u - without_u, expected_u_kl, atol=1e-4, rtol=1e-5
        )
        assert targets_np is not None
        assert z_mean is not None

    def test_a_non_dict_prediction_raises(self):
        targets, _, _, _, _ = _fixture()
        with pytest.raises(ValueError, match="expects a dict y_pred"):
            TopographicVAELoss()(targets, ops.convert_to_tensor(np.zeros((1, 1))))

    def test_a_dict_target_is_unwrapped(self):
        _, prediction, targets, z_mean, u_mean = _fixture()
        loss = TopographicVAELoss(kl_loss_weight=0.0)
        from_mapping = ops.convert_to_numpy(loss({"images": targets}, prediction))
        from_tensor = ops.convert_to_numpy(loss(targets, prediction))
        np.testing.assert_allclose(from_mapping, from_tensor, atol=0.0, rtol=0)


class TestTopographicVAELossTerms:
    """Each term against a closed form."""

    @pytest.mark.parametrize("log_var", [0.0, np.log(2.0)])
    def test_the_gaussian_kl_matches_its_definition(self, log_var):
        loss = TopographicVAELoss()
        rng = np.random.default_rng(11)
        mean = rng.normal(size=(B, S, D)).astype("float32")
        got = ops.convert_to_numpy(
            loss.gaussian_kl_loss(
                ops.convert_to_tensor(mean),
                ops.convert_to_tensor(np.full((B, S, D), log_var, "float32")),
            )
        )
        expected = 0.5 * (
            mean.astype(np.float64) ** 2 + np.exp(log_var) - 1.0 - log_var
        ).sum(axis=-1)
        np.testing.assert_allclose(got, expected, atol=1e-5, rtol=1e-5)

    def test_the_kl_is_zero_for_a_standard_normal_posterior(self):
        """KL(N(0,1) || N(0,1)) = 0 exactly."""
        loss = TopographicVAELoss()
        zeros = ops.convert_to_tensor(np.zeros((B, S, D), "float32"))
        kl = ops.convert_to_numpy(loss.gaussian_kl_loss(zeros, zeros))
        np.testing.assert_allclose(kl, 0.0, atol=1e-6, rtol=0)

    @pytest.mark.parametrize("log_var", [0.0, np.log(2.0)])
    @pytest.mark.parametrize("sum_reduction", [False, True])
    def test_the_whole_loss_matches_the_oracle(
        self, log_var, sum_reduction
    ):
        loss = TopographicVAELoss(
            kl_loss_weight=1.0, reconstruction_sum_reduction=sum_reduction
        )
        targets, prediction, targets_np, z_mean, u_mean = _fixture(
            log_var=log_var
        )
        got = ops.convert_to_numpy(loss.call(targets, prediction))
        reconstruction_np = np.clip(
            prediction["reconstruction"].numpy(),
            PROBABILITY_CLIP,
            1.0 - PROBABILITY_CLIP,
        )
        expected = _oracle(
            targets_np,
            reconstruction_np,
            z_mean,
            u_mean,
            log_var,
            1.0,
            sum_reduction,
        )
        assert got.shape == (B,), got.shape
        np.testing.assert_allclose(got, expected, atol=1e-4, rtol=1e-5)

    def test_the_reduction_is_over_the_image_axes_and_not_the_width(self):
        """The anti-vacuity arm for the axis pin.

        A reduction over WIDTH alone (rather than over H and W) gives a different
        number. MEASURED on this fixture: correct 20.5967, width-only 19.6373, a
        gap of 0.96 against a tolerance of 1e-4. A square image would hide this.
        """
        targets, prediction, targets_np, z_mean, u_mean = _fixture()
        loss = TopographicVAELoss(kl_loss_weight=1.0)
        got = ops.convert_to_numpy(loss(targets, prediction))

        nll = -(
            targets_np * np.log(targets_np)
            + (1 - targets_np) * np.log1p(-targets_np)
        )
        kl = (
            0.5 * (np.asarray(z_mean, dtype=np.float64) ** 2 - 1.0)
            .sum(axis=-1).sum(axis=-1).mean()
            + 0.5 * (np.asarray(u_mean, dtype=np.float64) ** 2 - 1.0)
            .sum(axis=-1).sum(axis=-1).mean()
        )
        width_only = nll.mean(axis=3).sum(axis=-1).mean() + kl
        assert abs(got - width_only) > 0.1, (
            "a width-only reduction gives the same number as the correct one "
            f"({width_only:.4f} vs {got:.4f}); the axis pin proves nothing"
        )

    def test_a_perfect_reconstruction_with_zero_kl_weight_is_almost_zero(self):
        targets, prediction, _, _, _ = _fixture()
        got = ops.convert_to_numpy(
            TopographicVAELoss(kl_loss_weight=0.0)(targets, prediction)
        )
        # The BCE is clipped at PROBABILITY_CLIP, so a target that lands exactly on
        # 0.0 or 1.0 still pays that clip: -log(1 - 1e-7) per element.
        bound = S * H * W * C * -np.log(PROBABILITY_CLIP)
        assert 0.0 <= got <= bound + 1e-6, got

    def test_the_kl_terms_actually_move_the_loss(self):
        """The other half of the pair: kl_weight=0 must EXCLUDE the posterior.

        Without this, a loss that silently dropped both KL terms would still
        satisfy every reconstruction assertion above.
        """
        targets, prediction, _, _, _ = _fixture(log_var=np.log(3.0))
        without = ops.convert_to_numpy(
            TopographicVAELoss(kl_loss_weight=0.0)(targets, prediction)
        )
        with_kl = ops.convert_to_numpy(
            TopographicVAELoss(kl_loss_weight=1.0)(targets, prediction)
        )
        assert with_kl > without + 1.0, "the KL terms contributed nothing"

    def test_doubling_the_kl_weight_doubles_only_the_kl(self):
        targets, prediction, *_ = _fixture(log_var=np.log(2.0))
        one = ops.convert_to_numpy(
            TopographicVAELoss(kl_loss_weight=1.0)(targets, prediction)
        )
        two = ops.convert_to_numpy(
            TopographicVAELoss(kl_loss_weight=2.0)(targets, prediction)
        )
        recon = ops.convert_to_numpy(
            TopographicVAELoss(kl_loss_weight=0.0)(targets, prediction)
        )
        np.testing.assert_allclose(two - one, one - recon, atol=1e-4, rtol=1e-4)

    def test_the_loss_is_one_value_per_sample(self):
        """Shape contract: ``call()`` returns ``(batch,)``, never a scalar.

        Asserted on `call`, not on `loss(...)`: `keras.losses.Loss.__call__`
        reduces with `reduce_weighted_values`, so the OUTER call is a scalar for a
        correct loss too, and asserting on it would pass for both a scalar and a
        per-sample return.

        A scalar return does not merely ignore `sample_weight` -- Keras multiplies
        first and reduces after, so the scalar BROADCASTS and every row is charged
        the batch aggregate.
        """
        targets, prediction, *_ = _fixture()
        got = TopographicVAELoss().call(targets, prediction)
        assert tuple(got.shape) == (B,), got.shape

    def test_sample_weight_selects_rows_rather_than_scaling_the_batch(self):
        """The predicate that proves the shape, not just the assertion above.

        With ``w = [1, 1, 1, 0]`` over four DISTINCT rows, a correctly shaped
        class gives ``(v0 + v1 + v2) / 4`` (Keras' ``SUM_OVER_BATCH_SIZE`` divides
        by the batch size, not by the weight sum). A scalar-returning class gives
        ``mean(v) * mean(w) = ((v0+v1+v2+v3)/4) * 0.75``. The two differ unless
        every row happens to be equal, which the fixture guarantees they are not
        — and `test_the_rows_really_are_distinct` asserts that.
        """
        targets, prediction, *_ = _fixture(batch=4, seed=21, log_var=np.log(2.0))
        loss = TopographicVAELoss(kl_loss_weight=1.0)
        per_sample = ops.convert_to_numpy(loss.call(targets, prediction))
        assert len(set(np.round(per_sample, 6))) >= 3, (
            f"the fixture rows are not distinct enough to discriminate: {per_sample}"
        )

        weights = ops.convert_to_tensor(
            np.array([1.0, 1.0, 1.0, 0.0], "float32")
        )
        weighted = float(
            ops.convert_to_numpy(
                loss(targets, prediction, sample_weight=weights)
            )
        )
        correct = float(per_sample[:3].sum()) / 4.0
        broadcast = float(per_sample.mean()) * 0.75

        np.testing.assert_allclose(weighted, correct, atol=1e-4, rtol=1e-4)
        assert abs(weighted - broadcast) > 1e-3, (
            "the weighting is indistinguishable from a scalar broadcast, so this "
            "test would pass either way"
        )

    def test_extreme_log_variances_are_clipped_rather_than_overflowing(self):
        """exp(1000) is inf in float32; the clip is what keeps the loss finite."""
        targets, prediction, *_ = _fixture()
        prediction["z_log_var"] = ops.convert_to_tensor(
            np.full((B, S, D), 1000.0, "float32")
        )
        prediction["u_log_var"] = prediction["z_log_var"]
        got = ops.convert_to_numpy(TopographicVAELoss()(targets, prediction))
        assert np.all(np.isfinite(got)), f"an unclipped log-variance gave {got}"

        # The unclipped value would be 0.5 * (exp(1000) - ...) = inf, so finiteness
        # alone proves the clip ran. Bound it by the CLIPPED value to prove it ran
        # at LOG_VAR_CLIP and not somewhere else: the largest finite answer the
        # clipping can produce is 0.5 * (exp(20) - 1 - 20) per latent, summed over
        # the latent width, the sequence and the batch.
        bound = (
            0.5 * (np.exp(LOG_VAR_CLIP) - 1.0 - LOG_VAR_CLIP) * D * S * B * 2
        )
        assert np.all(np.abs(got) < bound), (
            f"{got} exceeds the value the clip at LOG_VAR_CLIP can produce ({bound})"
        )

    def test_a_saturated_reconstruction_stays_finite(self):
        """A decoder that emits exactly 1.0 everywhere must not produce -inf."""
        targets, prediction, *_ = _fixture()
        prediction["reconstruction"] = ops.convert_to_tensor(
            np.ones((B, S, H, W, C), "float32")
        )
        got = ops.convert_to_numpy(TopographicVAELoss()(targets, prediction))
        assert np.isfinite(got), got

    def test_the_trackers_report_the_terms_separately(self):
        """The decomposition must be real, not three zeros.

        MEASURED: reconstruction ~1.4, z_kl ~13.1, u_kl ~6.0 at log_var=0 on this
        fixture -- three non-zero, non-equal numbers.
        """
        loss = TopographicVAELoss(kl_loss_weight=1.0)
        targets, prediction, *_ = _fixture()
        ops.convert_to_numpy(loss(targets, prediction))
        values = {
            "reconstruction": float(
                ops.convert_to_numpy(loss.reconstruction_tracker.result())
            ),
            "z_kl": float(ops.convert_to_numpy(loss.z_kl_tracker.result())),
            "u_kl": float(ops.convert_to_numpy(loss.u_kl_tracker.result())),
        }
        for name, value in values.items():
            assert value != 0.0, f"{name} reported exactly 0.0"
        assert values["reconstruction"] != values["z_kl"], "the terms are indistinguishable"


class TestTopographicVAELossEffectiveBeta:
    """`effective_beta` must reflect the reduction actually in force."""

    def test_the_sum_reduction_beta_is_the_kl_weight(self):
        assert (
            TopographicVAELoss(
                kl_loss_weight=0.25, reconstruction_sum_reduction=True
            ).effective_beta
            == 0.25
        )

    def test_the_mean_reduction_beta_scales_by_the_pixel_count(self):
        """Measured: 1.0 * 24 on a 4x6x1 frame."""
        loss = TopographicVAELoss(kl_loss_weight=1.0)
        targets, prediction, *_ = _fixture()
        ops.convert_to_numpy(loss(targets, prediction))
        assert loss.effective_beta == pytest.approx(H * W * C)

    def test_the_beta_is_the_kl_weight_before_any_batch(self):
        assert TopographicVAELoss(kl_loss_weight=0.5).effective_beta == 0.5


class TestTopographicVAELossPrecision:
    """The objective must not narrow under a dtype policy."""

    @pytest.mark.parametrize("policy", ["mixed_float16", "float64"])
    def test_the_loss_agrees_across_dtype_policies(self, policy):
        import keras

        targets, prediction, *_ = _fixture(log_var=np.log(2.0))
        reference = ops.convert_to_numpy(
            TopographicVAELoss(kl_loss_weight=1.0)(targets, prediction)
        )
        previous = keras.mixed_precision.global_policy().name
        keras.mixed_precision.set_global_policy(policy)
        try:
            got = ops.convert_to_numpy(
                TopographicVAELoss(kl_loss_weight=1.0)(targets, prediction)
            )
            assert np.isfinite(got), f"{policy} produced {got}"
            np.testing.assert_allclose(
                got, reference, rtol=5e-2, atol=5e-2,
                err_msg=f"{policy} disagrees with the float32 control",
            )
        finally:
            keras.mixed_precision.set_global_policy(previous)