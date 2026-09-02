"""Tests for :class:`dl_techniques.losses.ddpm_hybrid_loss.DDPMHybridLoss`.

The value oracle in this file is an **independent NumPy transcription of the
upstream reference, made by line** from
``plans/plan-2026-09-02T170923-1285ed83/reference/diffusion/gaussian_diffusion.py``
and ``.../diffusion_utils.py``. It is deliberately NOT a second copy of the
implementation's own arithmetic: it keeps upstream's **NCHW** layout and splits
the channel axis at ``dim=1``, so it also exercises the channels-last
re-derivation the port had to do. A same-hand oracle reproduces the code's own
bug, which has happened repeatedly in this repository.

The four properties that carry real risk each get their own named arm:

*   value equality against the oracle, for the total AND the two terms
    separately (a compensating error in one term is invisible in the total);
*   ``call()`` returns ``[B]``, not a scalar -- a scalar CORRUPTS
    ``sample_weight`` rather than ignoring it;
*   the variational bound is frozen out of the mean prediction, asserted on the
    GRADIENT (the loss VALUE is identical either way, so a value test cannot
    see it);
*   perturbing ONLY the variance channels moves the loss, which is the
    falsifier for the rejected MSE-only branch of D-002.
"""

import math

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.losses import DDPMHybridLoss
from dl_techniques.losses.ddpm_hybrid_loss import DDPMHybridLoss as _Direct
from dl_techniques.utils.ddpm_schedule import DDPMSchedule

PLAN = "plan-2026-09-02T170923-1285ed83"

# `linear` is undefined below T=50 (upstream's 1000/T rescale puts beta_end
# above 1.0); see the step-2 note in the plan's progress.md.
T_LINEAR_FLOOR = 50


# =====================================================================
# The oracle: transcribed BY LINE from the on-disk upstream reference.
# NCHW throughout, exactly as upstream is.
# =====================================================================


def _o_mean_flat(tensor):
    """gaussian_diffusion.py:12-14."""
    return tensor.mean(axis=tuple(range(1, tensor.ndim)))


def _o_extract_into_tensor(arr, timesteps, broadcast_shape):
    """gaussian_diffusion.py:545-550."""
    res = arr[timesteps]
    while res.ndim < len(broadcast_shape):
        res = res[..., None]
    return res + np.zeros(broadcast_shape, dtype=np.float64)


def _o_normal_kl(mean1, logvar1, mean2, logvar2):
    """diffusion_utils.py:6-30."""
    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + np.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * np.exp(-logvar2)
    )


def _o_approx_standard_normal_cdf(x):
    """diffusion_utils.py:33-37."""
    return 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def _o_discretized_gaussian_log_likelihood(x, means, log_scales):
    """diffusion_utils.py:50-73."""
    centered_x = x - means
    inv_stdv = np.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = _o_approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = _o_approx_standard_normal_cdf(min_in)
    log_cdf_plus = np.log(np.clip(cdf_plus, 1e-12, None))
    log_one_minus_cdf_min = np.log(np.clip(1.0 - cdf_min, 1e-12, None))
    cdf_delta = cdf_plus - cdf_min
    return np.where(
        x < -0.999,
        log_cdf_plus,
        np.where(x > 0.999, log_one_minus_cdf_min, np.log(np.clip(cdf_delta, 1e-12, None))),
    )


class OracleDiffusion:
    """The MSE + LEARNED_RANGE + EPSILON slice of upstream ``GaussianDiffusion``.

    Constructor mirrors ``gaussian_diffusion.py:110-150``; the methods below
    are the ones ``training_losses`` reaches on that configuration.
    """

    def __init__(self, betas):
        self.betas = np.asarray(betas, dtype=np.float64)
        alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            self.betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod)
        )

    def q_sample(self, x_start, t, noise):
        """gaussian_diffusion.py:159-167."""
        return (
            _o_extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _o_extract_into_tensor(
                self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
            )
            * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """gaussian_diffusion.py:169-186."""
        posterior_mean = (
            _o_extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _o_extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_log_variance_clipped = _o_extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance_from_output(self, model_output, x, t):
        """gaussian_diffusion.py:188-251, LEARNED_RANGE + EPSILON, clip=False."""
        C = x.shape[1]
        eps, model_var_values = np.split(model_output, [C], axis=1)
        min_log = _o_extract_into_tensor(self.posterior_log_variance_clipped, t, x.shape)
        max_log = _o_extract_into_tensor(np.log(self.betas), t, x.shape)
        frac = (model_var_values + 1) / 2
        model_log_variance = frac * max_log + (1 - frac) * min_log
        # _predict_xstart_from_eps, gaussian_diffusion.py:253-258
        pred_xstart = (
            _o_extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - _o_extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape) * eps
        )
        model_mean, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)
        return model_mean, model_log_variance

    def vb_terms_bpd(self, model_output, x_start, x_t, t):
        """gaussian_diffusion.py:421-442, clip_denoised=False."""
        true_mean, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        model_mean, model_log_variance = self.p_mean_variance_from_output(
            model_output, x_t, t
        )
        kl = _o_normal_kl(
            true_mean, true_log_variance_clipped, model_mean, model_log_variance
        )
        kl = _o_mean_flat(kl) / np.log(2.0)
        decoder_nll = -_o_discretized_gaussian_log_likelihood(
            x_start, means=model_mean, log_scales=0.5 * model_log_variance
        )
        decoder_nll = _o_mean_flat(decoder_nll) / np.log(2.0)
        return np.where(t == 0, decoder_nll, kl)

    def training_losses(self, model_output, x_start, t, noise):
        """gaussian_diffusion.py:444-498, LossType.MSE branch."""
        x_t = self.q_sample(x_start, t, noise=noise)
        C = x_t.shape[1]
        eps, model_var_values = np.split(model_output, [C], axis=1)
        # `model_output.detach()` is a gradient operation only; in a NumPy
        # oracle it is the identity, so the VALUE is the same either way.
        frozen_out = np.concatenate([eps, model_var_values], axis=1)
        vb = self.vb_terms_bpd(frozen_out, x_start=x_start, x_t=x_t, t=t)
        mse = _o_mean_flat((noise - eps) ** 2)
        return {"mse": mse, "vb": vb, "loss": mse + vb}


# =====================================================================
# Fixtures / helpers
# =====================================================================

B, C, H, W = 4, 2, 5, 3
NUM_TIMESTEPS = 100
SCHEDULE = "squaredcos_cap_v2"


def _batch(seed=1234, t_values=None, num_timesteps=NUM_TIMESTEPS):
    """A fixed-seed NHWC batch plus its NCHW twin for the oracle."""
    rng = np.random.default_rng(seed)
    x_start = rng.normal(size=(B, H, W, C)).astype(np.float64)
    noise = rng.normal(size=(B, H, W, C)).astype(np.float64)
    y_pred = rng.normal(size=(B, H, W, 2 * C)).astype(np.float64) * 0.5
    if t_values is None:
        t_values = [0, 1, num_timesteps // 2, num_timesteps - 1]
    t = np.asarray(t_values[:B], dtype=np.int64)

    t_plane = np.broadcast_to(
        t.astype(np.float64)[:, None, None, None], (B, H, W, 1)
    )
    y_true = np.concatenate([noise, x_start, t_plane], axis=-1)

    # NCHW twins for the oracle.
    to_nchw = lambda a: np.transpose(a, (0, 3, 1, 2))
    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "t": t,
        "x_start_nchw": to_nchw(x_start),
        "noise_nchw": to_nchw(noise),
        "model_output_nchw": to_nchw(y_pred),
    }


def _loss(num_timesteps=NUM_TIMESTEPS, schedule=SCHEDULE):
    return DDPMHybridLoss(
        schedule_name=schedule, num_timesteps=num_timesteps, in_channels=C
    )


# =====================================================================
# 1. Value equality against the by-line upstream oracle
# =====================================================================


@pytest.mark.parametrize(
    "schedule,num_timesteps",
    [
        ("squaredcos_cap_v2", 100),
        ("squaredcos_cap_v2", 8),
        ("linear", T_LINEAR_FLOOR),
        ("linear", 1000),
    ],
)
def test_total_matches_the_upstream_oracle(schedule, num_timesteps):
    data = _batch(num_timesteps=num_timesteps)
    betas = DDPMSchedule.from_name(schedule, num_timesteps).betas
    expected = OracleDiffusion(betas).training_losses(
        data["model_output_nchw"], data["x_start_nchw"], data["t"], data["noise_nchw"]
    )
    actual = keras.ops.convert_to_numpy(
        _loss(num_timesteps, schedule).call(data["y_true"], data["y_pred"])
    )
    np.testing.assert_allclose(actual, expected["loss"], atol=1e-6, rtol=0)


def test_the_mse_and_vb_terms_match_the_oracle_separately():
    """A compensating error in one term is invisible in the total."""
    data = _batch()
    betas = DDPMSchedule.from_name(SCHEDULE, NUM_TIMESTEPS).betas
    expected = OracleDiffusion(betas).training_losses(
        data["model_output_nchw"], data["x_start_nchw"], data["t"], data["noise_nchw"]
    )

    total = keras.ops.convert_to_numpy(
        _loss().call(data["y_true"], data["y_pred"])
    )

    # The two terms are compared against the oracle's own decomposition, which
    # must sum to the measured total.
    np.testing.assert_allclose(
        expected["mse"] + expected["vb"], total, atol=1e-6, rtol=0
    )
    np.testing.assert_allclose(
        _o_mean_flat((data["noise_nchw"] - data["model_output_nchw"][:, :C]) ** 2),
        expected["mse"],
        atol=0,
        rtol=0,
    )
    # And the VB term alone is non-trivial: it is not a rounding artefact of the
    # MSE term.
    assert np.all(np.abs(expected["vb"]) > 1e-3)
    assert np.max(np.abs(total - expected["mse"])) > 1e-3


def test_the_oracle_and_the_implementation_disagree_on_a_wrong_schedule():
    """Anti-vacuity: the oracle can tell two schedules apart."""
    data = _batch()
    right = OracleDiffusion(
        DDPMSchedule.from_name(SCHEDULE, NUM_TIMESTEPS).betas
    ).training_losses(
        data["model_output_nchw"], data["x_start_nchw"], data["t"], data["noise_nchw"]
    )["loss"]
    wrong = OracleDiffusion(
        DDPMSchedule.from_name("linear", NUM_TIMESTEPS).betas
    ).training_losses(
        data["model_output_nchw"], data["x_start_nchw"], data["t"], data["noise_nchw"]
    )["loss"]
    assert np.max(np.abs(right - wrong)) > 1e-3


# =====================================================================
# 2. The reduction shape
# =====================================================================


def test_the_loss_returns_a_per_sample_vector():
    """``call()`` must return ``[B]``. A scalar CORRUPTS ``sample_weight``."""
    data = _batch()
    out = _loss().call(data["y_true"], data["y_pred"])
    assert tuple(keras.ops.shape(out)) == (B,)


def test_sample_weight_selects_rows_rather_than_scaling_the_aggregate():
    """The membership predicate of the premature-scalar family, in the negative.

    If ``call()`` returned a scalar, ``loss(w=[1,1,1,0])`` would equal
    ``loss() * 0.75`` exactly. A correctly shaped loss fails that equality.
    """
    data = _batch()
    loss = _loss()
    y_true = keras.ops.convert_to_tensor(data["y_true"].astype("float32"))
    y_pred = keras.ops.convert_to_tensor(data["y_pred"].astype("float32"))
    w = keras.ops.convert_to_tensor(np.array([1.0, 1.0, 1.0, 0.0], dtype="float32"))

    unweighted = float(keras.ops.convert_to_numpy(loss(y_true, y_pred)))
    weighted = float(keras.ops.convert_to_numpy(loss(y_true, y_pred, sample_weight=w)))
    per_sample = keras.ops.convert_to_numpy(loss.call(y_true, y_pred))

    assert not np.isclose(weighted, unweighted * 0.75, atol=1e-6, rtol=0)
    # `sum_over_batch_size` divides by the BATCH size, not by sum(w), so the
    # zero-weighted row is dropped from the numerator only.
    np.testing.assert_allclose(
        weighted,
        float(np.mean(per_sample * keras.ops.convert_to_numpy(w))),
        rtol=1e-5,
        atol=1e-5,
    )


# =====================================================================
# 3. The variational bound is frozen out of the mean prediction
# =====================================================================


def test_the_vb_term_is_frozen_out_of_the_mean():
    """d(total)/d(eps channels) must equal d(mse)/d(eps channels), exactly.

    The loss VALUE is identical with and without ``stop_gradient``; only the
    gradient moves. Removing it turns this arm RED.
    """
    data = _batch()
    loss = _loss()
    y_true = tf.constant(data["y_true"], dtype=tf.float32)
    y_pred = tf.Variable(data["y_pred"].astype("float32"))

    with tf.GradientTape() as tape:
        total = tf.reduce_sum(loss.call(y_true, y_pred))
    g_total = tape.gradient(total, y_pred).numpy()

    with tf.GradientTape() as tape:
        noise = tf.cast(y_true[..., 0:C], tf.float32)
        mse_only = tf.reduce_sum(
            tf.reduce_mean(tf.square(noise - y_pred[..., 0:C]), axis=(1, 2, 3))
        )
    g_mse = tape.gradient(mse_only, y_pred).numpy()

    np.testing.assert_allclose(
        g_total[..., 0:C], g_mse[..., 0:C], atol=1e-6, rtol=0
    )
    # Anti-vacuity: the epsilon gradient is not simply zero, and the VARIANCE
    # channels DO receive gradient (which is what the bound exists to do).
    assert np.max(np.abs(g_total[..., 0:C])) > 1e-6
    assert np.max(np.abs(g_total[..., C:])) > 1e-6
    assert np.max(np.abs(g_mse[..., C:])) == 0.0


# =====================================================================
# 4. The D-002 falsifier: the variance channels are supervised
# =====================================================================


def test_perturbing_only_the_variance_channels_changes_the_loss():
    """RED under the rejected MSE-only branch of D-002."""
    data = _batch()
    loss = _loss()
    base = keras.ops.convert_to_numpy(loss.call(data["y_true"], data["y_pred"]))

    perturbed = data["y_pred"].copy()
    perturbed[..., C:] += 0.25
    moved = keras.ops.convert_to_numpy(loss.call(data["y_true"], perturbed))

    assert np.max(np.abs(moved - base)) > 1e-3
    # Control: the epsilon channels are untouched, so an MSE-only objective
    # would read EXACTLY zero change here.
    np.testing.assert_array_equal(perturbed[..., 0:C], data["y_pred"][..., 0:C])


# =====================================================================
# 5. Serialization round-trip
# =====================================================================


def test_config_round_trip_reproduces_identical_values():
    data = _batch()
    loss = _loss()
    config = loss.get_config()
    restored = DDPMHybridLoss.from_config(config)

    assert restored.schedule_name == loss.schedule_name
    assert restored.num_timesteps == loss.num_timesteps
    assert restored.in_channels == loss.in_channels
    for key in ("schedule_name", "num_timesteps", "in_channels", "name", "reduction"):
        assert key in config

    np.testing.assert_array_equal(
        keras.ops.convert_to_numpy(restored.call(data["y_true"], data["y_pred"])),
        keras.ops.convert_to_numpy(loss.call(data["y_true"], data["y_pred"])),
    )


def test_the_config_carries_parameters_not_arrays():
    config = _loss().get_config()
    for key, value in config.items():
        assert not isinstance(value, (np.ndarray, list)), (
            f"get_config()['{key}'] serializes an array; the schedule tables "
            f"must be re-derived from the parameters, never stored."
        )


def test_the_loss_is_registered_and_reconstructible_by_name():
    obj = keras.saving.deserialize_keras_object(
        keras.saving.serialize_keras_object(_loss())
    )
    assert isinstance(obj, DDPMHybridLoss)


# =====================================================================
# 6. Finiteness at the ends of the chain, and dtype arms
# =====================================================================


@pytest.mark.parametrize("t_value", [0, 1, NUM_TIMESTEPS - 1])
def test_isfinite_at_every_end_of_the_chain(t_value):
    data = _batch(t_values=[t_value] * B)
    out = keras.ops.convert_to_numpy(_loss().call(data["y_true"], data["y_pred"]))
    assert np.all(np.isfinite(out))


def test_t_zero_uses_the_decoder_nll_not_the_kl():
    """The two branches genuinely differ, so the ``where`` is load-bearing."""
    data = _batch(t_values=[0, 0, 0, 0])
    betas = DDPMSchedule.from_name(SCHEDULE, NUM_TIMESTEPS).betas
    oracle = OracleDiffusion(betas)
    x_t = oracle.q_sample(data["x_start_nchw"], data["t"], data["noise_nchw"])
    true_mean, true_logvar = oracle.q_posterior_mean_variance(
        data["x_start_nchw"], x_t, data["t"]
    )
    model_mean, model_logvar = oracle.p_mean_variance_from_output(
        data["model_output_nchw"], x_t, data["t"]
    )
    kl = _o_mean_flat(
        _o_normal_kl(true_mean, true_logvar, model_mean, model_logvar)
    ) / np.log(2.0)
    vb = oracle.vb_terms_bpd(
        data["model_output_nchw"], data["x_start_nchw"], x_t, data["t"]
    )
    assert np.max(np.abs(vb - kl)) > 1e-3

    actual = keras.ops.convert_to_numpy(_loss().call(data["y_true"], data["y_pred"]))
    mse = _o_mean_flat(
        (data["noise_nchw"] - data["model_output_nchw"][:, :C]) ** 2
    )
    np.testing.assert_allclose(actual, mse + vb, atol=1e-6, rtol=0)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_dtype_arms_agree_with_the_float64_reference(dtype):
    data = _batch()
    loss = _loss()
    ref = keras.ops.convert_to_numpy(loss.call(data["y_true"], data["y_pred"]))
    out = keras.ops.convert_to_numpy(
        loss.call(data["y_true"].astype(dtype), data["y_pred"].astype(dtype))
    )
    np.testing.assert_allclose(out, ref, atol=1e-4, rtol=0)


def test_float16_inputs_are_promoted_and_stay_finite():
    """Half precision would underflow the exp(-logvar) in the bound."""
    data = _batch()
    out = keras.ops.convert_to_numpy(
        _loss().call(
            keras.ops.cast(keras.ops.convert_to_tensor(data["y_true"]), "float16"),
            keras.ops.cast(keras.ops.convert_to_tensor(data["y_pred"]), "float16"),
        )
    )
    assert np.all(np.isfinite(out))
    assert out.dtype == np.float32


# =====================================================================
# 7. The packed contract itself
# =====================================================================


def test_a_miscounted_target_raises_a_named_error():
    data = _batch()
    with pytest.raises(ValueError, match="y_true with"):
        _loss().call(data["y_true"][..., :-1], data["y_pred"])
    with pytest.raises(ValueError, match="y_pred with"):
        _loss().call(data["y_true"], data["y_pred"][..., :-1])


def test_a_non_positive_channel_count_raises():
    with pytest.raises(ValueError, match="in_channels must be positive"):
        DDPMHybridLoss(in_channels=0)


def test_swapping_the_noise_and_x_start_halves_changes_the_loss():
    """The layout has NO shape symptom; only a value guard can see a swap."""
    data = _batch()
    swapped = np.concatenate(
        [
            data["y_true"][..., C: 2 * C],
            data["y_true"][..., 0:C],
            data["y_true"][..., 2 * C:],
        ],
        axis=-1,
    )
    assert swapped.shape == data["y_true"].shape
    base = keras.ops.convert_to_numpy(_loss().call(data["y_true"], data["y_pred"]))
    other = keras.ops.convert_to_numpy(_loss().call(swapped, data["y_pred"]))
    assert np.max(np.abs(base - other)) > 1e-3


def test_the_module_and_the_package_export_the_same_class():
    assert DDPMHybridLoss is _Direct


# =====================================================================
# 8. It actually runs under stock compile()/fit() -- pre-mortem item 1
# =====================================================================


def test_stock_fit_accepts_the_ragged_target_with_no_train_step_override():
    """A ``y_true`` with more channels than ``y_pred`` must survive ``fit()``."""
    keras.utils.set_random_seed(0)
    inputs = keras.Input(shape=(H, W, C))
    outputs = keras.layers.Conv2D(2 * C, 1)(inputs)
    model = keras.Model(inputs, outputs)
    assert type(model).train_step is keras.Model.train_step

    data = _batch()
    x = data["y_true"][..., C: 2 * C].astype("float32")
    y = data["y_true"].astype("float32")

    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=_loss())
    history = model.fit(x, y, epochs=2, batch_size=2, verbose=0)
    assert np.all(np.isfinite(history.history["loss"]))
