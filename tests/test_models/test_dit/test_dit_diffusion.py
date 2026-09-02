"""``GaussianDiffusion``: value equality against a transcribed oracle, and the
three silent-failure mechanisms.

The oracle in this file is a NumPy transcription of
``plans/.../reference/diffusion/gaussian_diffusion.py`` **by line**, with the
cited line ranges beside each function. It is not a second derivation from the
DDPM/DDIM papers, and it is not a re-statement of ``diffusion.py``'s own
arithmetic in different variable names -- either of those would be a
self-referential oracle that agrees with the code because both were written by
the same hand in the same hour. The single deliberate departure from the
reference text is the split axis: upstream splits the model output on ``dim=1``
because it is NCHW, this port is channels-LAST and splits on ``axis=-1``. That
one substitution is marked at the line it happens.

The constant TABLES are not re-derived here. They come from
:class:`~dl_techniques.utils.ddpm_schedule.DDPMSchedule`, which has its own
by-line oracle in ``tests/test_utils/test_ddpm_schedule.py``. What is under test
in this file is the sampler's algebra on top of them.

Three mechanisms here change no shape, no dtype and nothing about finiteness,
and are therefore invisible to every conventional arm:

1. **The ``nonzero_mask`` at ``t == 0``** -- dropping it adds one extra noise
   draw at the final step (``TestTZeroAddsNoNoise``).
2. **``sigma`` at ``eta = 0``** -- if it is not exactly zero, DDIM stops being
   deterministic (``TestDDIMAtEtaZeroIsDeterministic``).
3. **The respacing remap** -- feeding a model trained on 1000 steps the respaced
   index instead of the original one (``TestTheRespacingRemapIsReal``). That arm
   uses a recording stub and asserts the two candidate values are DIFFERENT
   before asserting which one arrived, so it cannot pass vacuously.

Reproducibility is proven with an explicit ``seed`` argument, never with
``keras.utils.set_random_seed``: that function does NOT re-seed an already-created
global ``SeedGenerator`` in this Keras version, so an arm written against it
would be measuring nothing. ``TestSeedingIsExplicit`` pins both halves of that.
"""

from typing import Any, Dict, List, Optional, Tuple

import keras
import numpy as np
import pytest

from dl_techniques.losses.ddpm_hybrid_loss import (
    DDPMHybridLoss,
    _discretized_gaussian_log_likelihood,
    _mean_flat,
    _normal_kl,
)
from dl_techniques.models.vision_language.dit.config import DiffusionConfig
from dl_techniques.models.vision_language.dit.diffusion import (
    DEFAULT_CLIP_DENOISED,
    GaussianDiffusion,
    MODEL_MEAN_TYPES,
    MODEL_VAR_TYPES,
)
from dl_techniques.models.vision_language.dit.model import DiT
from dl_techniques.utils.ddpm_schedule import DDPMSchedule

# ---------------------------------------------------------------------
# Shared fixtures / configuration
# ---------------------------------------------------------------------

#: A short chain, so a whole loop is cheap, and long enough that the respaced
#: indices and the original indices are numerically distinguishable.
ORIGINAL_STEPS: int = 40

#: The smallest DiT that still has a real block stack and a 2x2 patch grid.
TINY_DIT: Dict[str, Any] = {
    "input_size": 4,
    "patch_size": 2,
    "in_channels": 4,
    "hidden_size": 16,
    "depth": 1,
    "num_heads": 2,
    "mlp_ratio": 2.0,
    "class_dropout_rate": 0.1,
    "num_classes": 5,
    "learn_sigma": True,
}

TOL: Dict[str, float] = {"atol": 1e-6, "rtol": 0.0}


def _np(x: Any) -> np.ndarray:
    """Convert any backend tensor to a NumPy array.

    :param x: Tensor or array.
    :type x: Any
    :return: NumPy array.
    :rtype: np.ndarray
    """
    return np.asarray(keras.ops.convert_to_numpy(x))


# ---------------------------------------------------------------------
# The oracle -- transcribed from reference/diffusion/gaussian_diffusion.py
# ---------------------------------------------------------------------


def oracle_extract(arr: np.ndarray, t: np.ndarray, broadcast_shape) -> np.ndarray:
    """``_extract_into_tensor``, reference lines 545-550.

    :param arr: 1-D table.
    :type arr: np.ndarray
    :param t: Integer index per sample.
    :type t: np.ndarray
    :param broadcast_shape: Shape to broadcast to.
    :return: float64 array of ``broadcast_shape``.
    :rtype: np.ndarray
    """
    res = np.asarray(arr, dtype=np.float64)[np.asarray(t)]
    while res.ndim < len(broadcast_shape):
        res = res[..., None]
    return res + np.zeros(broadcast_shape, dtype=np.float64)


def oracle_q_sample(
    sched: DDPMSchedule, x_start: np.ndarray, t: np.ndarray, noise: np.ndarray
) -> np.ndarray:
    """``q_sample``, reference lines 159-167.

    :param sched: The schedule.
    :type sched: DDPMSchedule
    :param x_start: Clean data.
    :type x_start: np.ndarray
    :param t: Timesteps.
    :type t: np.ndarray
    :param noise: Standard-normal noise.
    :type noise: np.ndarray
    :return: ``x_t``.
    :rtype: np.ndarray
    """
    return (
        oracle_extract(sched.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        + oracle_extract(sched.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        * noise
    )


def oracle_q_posterior(
    sched: DDPMSchedule, x_start: np.ndarray, x_t: np.ndarray, t: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``q_posterior_mean_variance``, reference lines 169-186.

    :param sched: The schedule.
    :type sched: DDPMSchedule
    :param x_start: Clean data.
    :type x_start: np.ndarray
    :param x_t: Noised data.
    :type x_t: np.ndarray
    :param t: Timesteps.
    :type t: np.ndarray
    :return: ``(mean, variance, log_variance_clipped)``.
    :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]
    """
    posterior_mean = (
        oracle_extract(sched.posterior_mean_coef1, t, x_t.shape) * x_start
        + oracle_extract(sched.posterior_mean_coef2, t, x_t.shape) * x_t
    )
    posterior_variance = oracle_extract(sched.posterior_variance, t, x_t.shape)
    posterior_log_variance_clipped = oracle_extract(
        sched.posterior_log_variance_clipped, t, x_t.shape
    )
    return posterior_mean, posterior_variance, posterior_log_variance_clipped


def oracle_predict_xstart_from_eps(
    sched: DDPMSchedule, x_t: np.ndarray, t: np.ndarray, eps: np.ndarray
) -> np.ndarray:
    """``_predict_xstart_from_eps``, reference lines 253-258.

    :param sched: The schedule.
    :type sched: DDPMSchedule
    :param x_t: Noised data.
    :type x_t: np.ndarray
    :param t: Timesteps.
    :type t: np.ndarray
    :param eps: Predicted noise.
    :type eps: np.ndarray
    :return: Predicted ``x_0``.
    :rtype: np.ndarray
    """
    return (
        oracle_extract(sched.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
        - oracle_extract(sched.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
    )


def oracle_predict_eps_from_xstart(
    sched: DDPMSchedule, x_t: np.ndarray, t: np.ndarray, pred_xstart: np.ndarray
) -> np.ndarray:
    """``_predict_eps_from_xstart``, reference lines 260-263.

    :param sched: The schedule.
    :type sched: DDPMSchedule
    :param x_t: Noised data.
    :type x_t: np.ndarray
    :param t: Timesteps.
    :type t: np.ndarray
    :param pred_xstart: Predicted ``x_0``.
    :type pred_xstart: np.ndarray
    :return: Implied noise.
    :rtype: np.ndarray
    """
    return (
        oracle_extract(sched.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
        - pred_xstart
    ) / oracle_extract(sched.sqrt_recipm1_alphas_cumprod, t, x_t.shape)


def oracle_p_mean_variance(
    sched: DDPMSchedule,
    model_output: np.ndarray,
    x: np.ndarray,
    t: np.ndarray,
    var_type: str,
    clip_denoised: bool,
    mean_type: str = "epsilon",
) -> Dict[str, np.ndarray]:
    """``p_mean_variance``, reference lines 188-251.

    :param sched: The schedule.
    :type sched: DDPMSchedule
    :param model_output: What the model returned.
    :type model_output: np.ndarray
    :param x: ``x_t``.
    :type x: np.ndarray
    :param t: Timesteps.
    :type t: np.ndarray
    :param var_type: One of :data:`MODEL_VAR_TYPES`.
    :type var_type: str
    :param clip_denoised: Clamp ``x_0_hat`` to ``[-1, 1]``.
    :type clip_denoised: bool
    :param mean_type: One of :data:`MODEL_MEAN_TYPES`.
    :type mean_type: str
    :return: ``{'mean', 'variance', 'log_variance', 'pred_xstart'}``.
    :rtype: Dict[str, np.ndarray]
    """
    channels = x.shape[-1]

    if var_type in ("learned", "learned_range"):
        # reference:206 -- `th.split(model_output, C, dim=1)`. THE ONE
        # DEPARTURE: dim=1 is upstream's NCHW channel axis; ours is -1.
        model_output, model_var_values = (
            model_output[..., :channels],
            model_output[..., channels:],
        )
        if var_type == "learned":
            model_log_variance = model_var_values
            model_variance = np.exp(model_log_variance)
        else:
            min_log = oracle_extract(
                sched.posterior_log_variance_clipped, t, x.shape
            )
            max_log = oracle_extract(np.log(sched.betas), t, x.shape)
            frac = (model_var_values + 1) / 2
            model_log_variance = frac * max_log + (1 - frac) * min_log
            model_variance = np.exp(model_log_variance)
    else:
        # reference:217-224
        table = {
            "fixed_large": (
                np.append(sched.posterior_variance[1], sched.betas[1:]),
                np.log(np.append(sched.posterior_variance[1], sched.betas[1:])),
            ),
            "fixed_small": (
                sched.posterior_variance,
                sched.posterior_log_variance_clipped,
            ),
        }[var_type]
        model_variance = oracle_extract(table[0], t, x.shape)
        model_log_variance = oracle_extract(table[1], t, x.shape)

    def process_xstart(value: np.ndarray) -> np.ndarray:
        if clip_denoised:
            return np.clip(value, -1, 1)
        return value

    if mean_type == "start_x":
        pred_xstart = process_xstart(model_output)
    else:
        pred_xstart = process_xstart(
            oracle_predict_xstart_from_eps(sched, x_t=x, t=t, eps=model_output)
        )
    model_mean, _, _ = oracle_q_posterior(sched, pred_xstart, x, t)

    return {
        "mean": model_mean,
        "variance": model_variance,
        "log_variance": model_log_variance,
        "pred_xstart": pred_xstart,
    }


def oracle_p_sample(
    sched: DDPMSchedule,
    model_output: np.ndarray,
    x: np.ndarray,
    t: np.ndarray,
    noise: np.ndarray,
    var_type: str,
    clip_denoised: bool,
) -> np.ndarray:
    """``p_sample``, reference lines 283-295.

    :param sched: The schedule.
    :type sched: DDPMSchedule
    :param model_output: What the model returned.
    :type model_output: np.ndarray
    :param x: ``x_t``.
    :type x: np.ndarray
    :param t: Timesteps.
    :type t: np.ndarray
    :param noise: The (fixed) noise draw.
    :type noise: np.ndarray
    :param var_type: One of :data:`MODEL_VAR_TYPES`.
    :type var_type: str
    :param clip_denoised: Clamp ``x_0_hat``.
    :type clip_denoised: bool
    :return: ``x_{t-1}``.
    :rtype: np.ndarray
    """
    out = oracle_p_mean_variance(
        sched, model_output, x, t, var_type, clip_denoised
    )
    nonzero_mask = (np.asarray(t) != 0).astype(np.float64).reshape(
        -1, *([1] * (x.ndim - 1))
    )
    return out["mean"] + nonzero_mask * np.exp(0.5 * out["log_variance"]) * noise


def oracle_ddim_sample(
    sched: DDPMSchedule,
    model_output: np.ndarray,
    x: np.ndarray,
    t: np.ndarray,
    noise: np.ndarray,
    var_type: str,
    clip_denoised: bool,
    eta: float,
) -> np.ndarray:
    """``ddim_sample``, reference lines 334-364.

    :param sched: The schedule.
    :type sched: DDPMSchedule
    :param model_output: What the model returned.
    :type model_output: np.ndarray
    :param x: ``x_t``.
    :type x: np.ndarray
    :param t: Timesteps.
    :type t: np.ndarray
    :param noise: The (fixed) noise draw.
    :type noise: np.ndarray
    :param var_type: One of :data:`MODEL_VAR_TYPES`.
    :type var_type: str
    :param clip_denoised: Clamp ``x_0_hat``.
    :type clip_denoised: bool
    :param eta: DDIM stochasticity.
    :type eta: float
    :return: ``x_{t-1}``.
    :rtype: np.ndarray
    """
    out = oracle_p_mean_variance(
        sched, model_output, x, t, var_type, clip_denoised
    )
    eps = oracle_predict_eps_from_xstart(sched, x, t, out["pred_xstart"])
    alpha_bar = oracle_extract(sched.alphas_cumprod, t, x.shape)
    alpha_bar_prev = oracle_extract(sched.alphas_cumprod_prev, t, x.shape)
    sigma = (
        eta
        * np.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
        * np.sqrt(1 - alpha_bar / alpha_bar_prev)
    )
    mean_pred = out["pred_xstart"] * np.sqrt(alpha_bar_prev) + np.sqrt(
        1 - alpha_bar_prev - sigma ** 2
    ) * eps
    nonzero_mask = (np.asarray(t) != 0).astype(np.float64).reshape(
        -1, *([1] * (x.ndim - 1))
    )
    return mean_pred + nonzero_mask * sigma * noise


# ---------------------------------------------------------------------
# Instruments
# ---------------------------------------------------------------------


class ConstantModel:
    """A model callable that ignores its inputs and returns a fixed tensor.

    Used wherever the arm is about the sampler's algebra rather than about DiT.

    :param output: The tensor to return from every call.
    :type output: Any
    """

    def __init__(self, output: Any) -> None:
        self.output = output
        self.calls: int = 0

    def __call__(self, x: Any, t: Any, **kwargs: Any) -> Any:
        """Return the fixed output.

        :param x: Ignored.
        :type x: Any
        :param t: Ignored.
        :type t: Any
        :param kwargs: Ignored.
        :type kwargs: Any
        :return: The fixed output tensor.
        :rtype: Any
        """
        self.calls += 1
        return self.output


class RecordingModel:
    """A model callable that records every ``t`` it is handed.

    :param out_channels: Channel count of the returned tensor.
    :type out_channels: int
    """

    def __init__(self, out_channels: int) -> None:
        self.out_channels = out_channels
        self.seen_t: List[np.ndarray] = []

    def __call__(self, x: Any, t: Any, **kwargs: Any) -> Any:
        """Record ``t`` and return zeros of the declared channel count.

        :param x: The current state, used only for its shape.
        :type x: Any
        :param t: The timestep tensor, recorded.
        :type t: Any
        :param kwargs: Ignored.
        :type kwargs: Any
        :return: Zeros ``[B, H, W, out_channels]``.
        :rtype: Any
        """
        self.seen_t.append(_np(t).copy())
        shape = list(keras.ops.shape(x))
        shape[-1] = self.out_channels
        return keras.ops.zeros(tuple(shape), dtype=x.dtype)


def make_case(
    seed: int,
    batch: int = 3,
    size: int = 4,
    channels: int = 4,
    steps: int = ORIGINAL_STEPS,
    schedule_name: str = "squaredcos_cap_v2",
) -> Tuple[DDPMSchedule, np.ndarray, np.ndarray, np.ndarray]:
    """Build a deterministic ``(schedule, x, t, model_output)`` case.

    ``t`` deliberately contains ``0`` in the first slot so every arm that runs
    against it exercises the ``nonzero_mask`` branch as well.

    :param seed: NumPy seed.
    :type seed: int
    :param batch: Batch size (>= 2).
    :type batch: int
    :param size: Spatial side.
    :type size: int
    :param channels: Latent channels ``C``.
    :type channels: int
    :param steps: Chain length.
    :type steps: int
    :param schedule_name: Beta schedule name.
    :type schedule_name: str
    :return: ``(schedule, x, t, model_output)`` with ``model_output`` carrying
        ``2 * channels`` channels.
    :rtype: Tuple[DDPMSchedule, np.ndarray, np.ndarray, np.ndarray]
    """
    rng = np.random.default_rng(seed)
    sched = DDPMSchedule.from_name(schedule_name, steps)
    x = rng.normal(size=(batch, size, size, channels))
    t = np.concatenate(
        [np.array([0]), rng.integers(1, steps, size=batch - 1)]
    ).astype(np.int32)
    model_output = rng.normal(size=(batch, size, size, 2 * channels)) * 0.5
    return sched, x, t, model_output


# ---------------------------------------------------------------------
# Value equality against the oracle
# ---------------------------------------------------------------------


class TestPMeanVarianceMatchesTheOracle:
    """``p_mean_variance`` reproduces the transcribed reference, key by key."""

    @pytest.mark.parametrize("var_type", MODEL_VAR_TYPES)
    @pytest.mark.parametrize("clip_denoised", [False, True])
    def test_every_output_key(self, var_type: str, clip_denoised: bool) -> None:
        sched, x, t, model_output = make_case(seed=11)
        if var_type in ("fixed_small", "fixed_large"):
            # A learn_sigma=False model emits C channels, not 2C.
            model_output = model_output[..., : x.shape[-1]]

        gd = GaussianDiffusion(sched, model_var_type=var_type)
        got = gd.p_mean_variance(
            ConstantModel(keras.ops.convert_to_tensor(model_output)),
            keras.ops.convert_to_tensor(x),
            keras.ops.convert_to_tensor(t),
            clip_denoised=clip_denoised,
        )
        want = oracle_p_mean_variance(
            sched, model_output, x, t, var_type, clip_denoised
        )

        for key in ("mean", "variance", "log_variance", "pred_xstart"):
            np.testing.assert_allclose(
                _np(got[key]), want[key], err_msg=key, **TOL
            )
            assert np.isfinite(_np(got[key])).all(), key

    @pytest.mark.parametrize("mean_type", MODEL_MEAN_TYPES)
    def test_both_mean_types(self, mean_type: str) -> None:
        sched, x, t, model_output = make_case(seed=12)
        gd = GaussianDiffusion(
            sched, model_mean_type=mean_type, model_var_type="learned_range"
        )
        got = gd.p_mean_variance(
            ConstantModel(keras.ops.convert_to_tensor(model_output)),
            keras.ops.convert_to_tensor(x),
            keras.ops.convert_to_tensor(t),
        )
        want = oracle_p_mean_variance(
            sched,
            model_output,
            x,
            t,
            "learned_range",
            DEFAULT_CLIP_DENOISED,
            mean_type=mean_type,
        )
        np.testing.assert_allclose(_np(got["mean"]), want["mean"], **TOL)
        np.testing.assert_allclose(
            _np(got["pred_xstart"]), want["pred_xstart"], **TOL
        )

    def test_the_clip_actually_bites(self) -> None:
        """Anti-vacuity: the two ``clip_denoised`` arms must DISAGREE.

        Without this the clip arm above would be satisfied by a sampler that
        ignores the flag entirely.
        """
        sched, x, t, model_output = make_case(seed=13)
        gd = GaussianDiffusion(sched)
        args = (
            ConstantModel(keras.ops.convert_to_tensor(model_output)),
            keras.ops.convert_to_tensor(x),
            keras.ops.convert_to_tensor(t),
        )
        clipped = _np(gd.p_mean_variance(*args, clip_denoised=True)["pred_xstart"])
        raw = _np(gd.p_mean_variance(*args, clip_denoised=False)["pred_xstart"])
        assert np.abs(raw).max() > 1.0
        assert not np.allclose(clipped, raw)


class TestQSampleMatchesTheOracle:
    """``q_sample`` reproduces reference lines 159-167."""

    def test_values(self) -> None:
        sched, x_start, t, _ = make_case(seed=21)
        rng = np.random.default_rng(99)
        noise = rng.normal(size=x_start.shape)
        gd = GaussianDiffusion(sched)
        got = gd.q_sample(
            keras.ops.convert_to_tensor(x_start),
            keras.ops.convert_to_tensor(t),
            noise=keras.ops.convert_to_tensor(noise),
        )
        np.testing.assert_allclose(
            _np(got), oracle_q_sample(sched, x_start, t, noise), **TOL
        )


class TestQPosteriorMatchesTheOracle:
    """``q_posterior_mean_variance`` reproduces reference lines 169-186."""

    def test_values(self) -> None:
        sched, x_start, t, _ = make_case(seed=22)
        rng = np.random.default_rng(98)
        x_t = rng.normal(size=x_start.shape)
        gd = GaussianDiffusion(sched)
        mean, var, log_var = gd.q_posterior_mean_variance(
            keras.ops.convert_to_tensor(x_start),
            keras.ops.convert_to_tensor(x_t),
            keras.ops.convert_to_tensor(t),
        )
        w_mean, w_var, w_log = oracle_q_posterior(sched, x_start, x_t, t)
        np.testing.assert_allclose(_np(mean), w_mean, **TOL)
        np.testing.assert_allclose(_np(var), w_var, **TOL)
        np.testing.assert_allclose(_np(log_var), w_log, **TOL)


class TestOneStepMatchesTheOracle:
    """One ``p_sample`` / ``ddim_sample`` step at a FIXED noise draw."""

    def test_p_sample(self) -> None:
        sched, x, t, model_output = make_case(seed=31)
        rng = np.random.default_rng(1234)
        noise = rng.normal(size=x.shape)
        gd = GaussianDiffusion(sched)
        got = gd.p_sample(
            ConstantModel(keras.ops.convert_to_tensor(model_output)),
            keras.ops.convert_to_tensor(x),
            keras.ops.convert_to_tensor(t),
            noise=keras.ops.convert_to_tensor(noise),
        )
        want = oracle_p_sample(
            sched, model_output, x, t, noise, "learned_range",
            DEFAULT_CLIP_DENOISED,
        )
        np.testing.assert_allclose(_np(got["sample"]), want, **TOL)

    @pytest.mark.parametrize("eta", [0.0, 0.5, 1.0])
    def test_ddim_sample(self, eta: float) -> None:
        sched, x, t, model_output = make_case(seed=32)
        rng = np.random.default_rng(4321)
        noise = rng.normal(size=x.shape)
        gd = GaussianDiffusion(sched)
        got = gd.ddim_sample(
            ConstantModel(keras.ops.convert_to_tensor(model_output)),
            keras.ops.convert_to_tensor(x),
            keras.ops.convert_to_tensor(t),
            eta=eta,
            noise=keras.ops.convert_to_tensor(noise),
        )
        want = oracle_ddim_sample(
            sched, model_output, x, t, noise, "learned_range",
            DEFAULT_CLIP_DENOISED, eta,
        )
        np.testing.assert_allclose(_np(got["sample"]), want, **TOL)


# ---------------------------------------------------------------------
# The three silent-failure mechanisms
# ---------------------------------------------------------------------


class TestTZeroAddsNoNoise:
    """``t == 0`` must land exactly on the mean: the ``nonzero_mask``.

    RED if the mask is dropped. The noise draw here is deliberately LARGE so a
    dropped mask cannot hide inside a tolerance.
    """

    @staticmethod
    def _case() -> Tuple[GaussianDiffusion, np.ndarray, np.ndarray, np.ndarray, ConstantModel]:
        sched, x, _, model_output = make_case(seed=41, batch=4)
        t = np.array([0, 0, 5, 9], dtype=np.int32)
        noise = np.full(x.shape, 50.0)
        gd = GaussianDiffusion(sched)
        return gd, x, t, noise, ConstantModel(
            keras.ops.convert_to_tensor(model_output)
        )

    def test_p_sample_at_t_zero_equals_the_mean(self) -> None:
        gd, x, t, noise, model = self._case()
        args = (
            model,
            keras.ops.convert_to_tensor(x),
            keras.ops.convert_to_tensor(t),
        )
        out = gd.p_sample(*args, noise=keras.ops.convert_to_tensor(noise))
        mean = _np(gd.p_mean_variance(*args)["mean"])
        sample = _np(out["sample"])
        np.testing.assert_allclose(sample[t == 0], mean[t == 0], **TOL)
        # Anti-vacuity: the t != 0 rows MUST have moved.
        assert np.abs(sample[t != 0] - mean[t != 0]).min() > 0.0

    def test_ddim_sample_at_t_zero_has_no_noise(self) -> None:
        gd, x, t, noise, model = self._case()
        args = (
            model,
            keras.ops.convert_to_tensor(x),
            keras.ops.convert_to_tensor(t),
        )
        with_noise = _np(
            gd.ddim_sample(
                *args, eta=1.0, noise=keras.ops.convert_to_tensor(noise)
            )["sample"]
        )
        without_noise = _np(
            gd.ddim_sample(
                *args, eta=1.0, noise=keras.ops.zeros_like(
                    keras.ops.convert_to_tensor(noise)
                )
            )["sample"]
        )
        np.testing.assert_allclose(
            with_noise[t == 0], without_noise[t == 0], **TOL
        )
        assert np.abs(
            with_noise[t != 0] - without_noise[t != 0]
        ).min() > 0.0

    def test_the_loop_final_step_is_noise_free(self) -> None:
        """End-to-end: two runs of the same chain differing only in the LAST
        draw must agree, because the last draw is masked out."""
        sched = DDPMSchedule.from_name("squaredcos_cap_v2", 4)
        gd = GaussianDiffusion(sched)
        rng = np.random.default_rng(7)
        x_t = keras.ops.convert_to_tensor(rng.normal(size=(2, 4, 4, 4)))
        model = ConstantModel(
            keras.ops.convert_to_tensor(rng.normal(size=(2, 4, 4, 8)) * 0.3)
        )
        t0 = keras.ops.zeros((2,), dtype="int32")
        a = _np(gd.p_sample(model, x_t, t0, seed=1)["sample"])
        b = _np(gd.p_sample(model, x_t, t0, seed=999)["sample"])
        np.testing.assert_allclose(a, b, **TOL)


class TestDDIMAtEtaZeroIsDeterministic:
    """``eta = 0`` makes ``sigma`` exactly zero, so the noise cannot reach the
    output. RED if ``sigma`` is non-zero at ``eta = 0``."""

    def test_two_different_noise_draws_agree(self) -> None:
        sched, x, t, model_output = make_case(seed=51)
        gd = GaussianDiffusion(sched)
        model = ConstantModel(keras.ops.convert_to_tensor(model_output))
        rng = np.random.default_rng(0)
        n1 = keras.ops.convert_to_tensor(rng.normal(size=x.shape) * 100.0)
        n2 = keras.ops.convert_to_tensor(rng.normal(size=x.shape) * 100.0)
        xt = keras.ops.convert_to_tensor(x)
        tt = keras.ops.convert_to_tensor(t)
        a = _np(gd.ddim_sample(model, xt, tt, eta=0.0, noise=n1)["sample"])
        b = _np(gd.ddim_sample(model, xt, tt, eta=0.0, noise=n2)["sample"])
        np.testing.assert_allclose(a, b, atol=0.0, rtol=0.0)

    def test_a_nonzero_eta_does_move(self) -> None:
        """Anti-vacuity: the same comparison at ``eta = 1`` must DISAGREE, so
        the determinism above is a property of ``eta``, not of the plumbing."""
        sched, x, t, model_output = make_case(seed=51)
        gd = GaussianDiffusion(sched)
        model = ConstantModel(keras.ops.convert_to_tensor(model_output))
        rng = np.random.default_rng(0)
        n1 = keras.ops.convert_to_tensor(rng.normal(size=x.shape) * 100.0)
        n2 = keras.ops.convert_to_tensor(rng.normal(size=x.shape) * 100.0)
        xt = keras.ops.convert_to_tensor(x)
        tt = keras.ops.convert_to_tensor(t)
        a = _np(gd.ddim_sample(model, xt, tt, eta=1.0, noise=n1)["sample"])
        b = _np(gd.ddim_sample(model, xt, tt, eta=1.0, noise=n2)["sample"])
        assert not np.allclose(a[t != 0], b[t != 0])

    def test_the_whole_loop_is_deterministic_at_eta_zero(self) -> None:
        sched = DDPMSchedule.from_name("squaredcos_cap_v2", 6)
        gd = GaussianDiffusion(sched)
        rng = np.random.default_rng(3)
        x_t = keras.ops.convert_to_tensor(rng.normal(size=(2, 4, 4, 4)))
        model = ConstantModel(
            keras.ops.convert_to_tensor(rng.normal(size=(2, 4, 4, 8)) * 0.2)
        )
        a = _np(gd.ddim_sample_loop(model, noise=x_t, eta=0.0, seed=1))
        b = _np(gd.ddim_sample_loop(model, noise=x_t, eta=0.0, seed=987654))
        np.testing.assert_allclose(a, b, atol=0.0, rtol=0.0)


class TestTheRespacingRemapIsReal:
    """The model must be handed the ORIGINAL index, not the respaced one.

    RED if the remap is bypassed. Every arm first asserts the two candidate
    value sets DIFFER, so it cannot pass vacuously on a schedule where they
    happen to coincide.
    """

    @staticmethod
    def _respaced(steps: int = 5) -> GaussianDiffusion:
        return GaussianDiffusion.from_name(
            "squaredcos_cap_v2", ORIGINAL_STEPS, timestep_respacing=steps
        )

    def test_the_map_is_not_the_identity(self) -> None:
        gd = self._respaced()
        assert gd.timestep_map is not None
        assert gd.num_timesteps == 5
        assert gd.original_num_steps == ORIGINAL_STEPS
        respaced_indices = np.arange(gd.num_timesteps)
        assert not np.array_equal(gd.timestep_map, respaced_indices)

    def test_a_single_step_passes_the_original_index(self) -> None:
        gd = self._respaced()
        model = RecordingModel(out_channels=8)
        x = keras.ops.convert_to_tensor(
            np.zeros((2, 4, 4, 4), dtype=np.float32)
        )
        for respaced_index in range(gd.num_timesteps):
            model.seen_t = []
            gd.p_sample(
                model, x, keras.ops.full((2,), respaced_index, dtype="int32")
            )
            seen = model.seen_t[0]
            expected = int(gd.timestep_map[respaced_index])
            assert (seen == expected).all(), (
                f"respaced index {respaced_index} should reach the model as "
                f"{expected}, got {seen}"
            )

    def test_the_loop_passes_original_indices_in_order(self) -> None:
        gd = self._respaced()
        model = RecordingModel(out_channels=8)
        x = keras.ops.convert_to_tensor(
            np.zeros((2, 4, 4, 4), dtype=np.float32)
        )
        gd.p_sample_loop(model, noise=x, seed=0)
        seen = [int(v[0]) for v in model.seen_t]
        expected = list(reversed([int(v) for v in gd.timestep_map]))
        respaced = list(reversed(range(gd.num_timesteps)))
        assert seen != respaced, (
            "vacuous: the respaced indices and the mapped indices coincide"
        )
        assert seen == expected

    def test_ddim_loop_passes_original_indices(self) -> None:
        gd = self._respaced()
        model = RecordingModel(out_channels=8)
        x = keras.ops.convert_to_tensor(
            np.zeros((2, 4, 4, 4), dtype=np.float32)
        )
        gd.ddim_sample_loop(model, noise=x, seed=0)
        seen = [int(v[0]) for v in model.seen_t]
        expected = list(reversed([int(v) for v in gd.timestep_map]))
        assert seen != list(reversed(range(gd.num_timesteps)))
        assert seen == expected

    def test_an_unrespaced_process_passes_t_through(self) -> None:
        gd = GaussianDiffusion.from_name("squaredcos_cap_v2", 4)
        assert gd.timestep_map is None
        model = RecordingModel(out_channels=8)
        x = keras.ops.convert_to_tensor(
            np.zeros((2, 4, 4, 4), dtype=np.float32)
        )
        gd.p_sample(model, x, keras.ops.full((2,), 3, dtype="int32"))
        assert (model.seen_t[0] == 3).all()

    def test_the_tables_are_indexed_by_the_RESPACED_index(self) -> None:
        """The other half of the contract: the schedule the sampler reads is the
        SHORTENED one, so its tables have ``num_timesteps`` entries and index
        ``t`` directly. If both sides used the original index this would raise
        or silently read the wrong row."""
        gd = self._respaced()
        assert gd.schedule.betas.shape == (gd.num_timesteps,)
        model = RecordingModel(out_channels=8)
        x = keras.ops.convert_to_tensor(
            np.zeros((2, 4, 4, 4), dtype=np.float32)
        )
        out = gd.p_mean_variance(
            model, x, keras.ops.full((2,), gd.num_timesteps - 1, dtype="int32")
        )
        assert np.isfinite(_np(out["mean"])).all()


# ---------------------------------------------------------------------
# Loops against the real DiT
# ---------------------------------------------------------------------


def _tiny_dit() -> DiT:
    """Build the smallest DiT this file uses.

    :return: A constructed model.
    :rtype: DiT
    """
    keras.utils.set_random_seed(1234)
    return DiT(**TINY_DIT)


class TestLoopsAgainstTheRealModel:
    """A 2-step ancestral loop and a 2-step DDIM loop drive the real DiT."""

    @staticmethod
    def _plain_callable(model: DiT, labels: Any):
        def call(x: Any, t: Any, **kwargs: Any) -> Any:
            return model([x, t, labels], training=False)

        return call

    def test_p_sample_loop_two_steps(self) -> None:
        model = _tiny_dit()
        gd = GaussianDiffusion.from_name(
            "squaredcos_cap_v2", ORIGINAL_STEPS, timestep_respacing=2
        )
        labels = keras.ops.convert_to_tensor(np.array([0, 1], dtype="int32"))
        out = gd.p_sample_loop(
            self._plain_callable(model, labels), shape=(2, 4, 4, 4), seed=5
        )
        assert tuple(keras.ops.shape(out)) == (2, 4, 4, 4)
        assert np.isfinite(_np(out)).all()

    def test_ddim_sample_loop_two_steps(self) -> None:
        model = _tiny_dit()
        gd = GaussianDiffusion.from_name(
            "squaredcos_cap_v2", ORIGINAL_STEPS, timestep_respacing=2
        )
        labels = keras.ops.convert_to_tensor(np.array([0, 1], dtype="int32"))
        out = gd.ddim_sample_loop(
            self._plain_callable(model, labels),
            shape=(2, 4, 4, 4),
            eta=0.0,
            seed=5,
        )
        assert tuple(keras.ops.shape(out)) == (2, 4, 4, 4)
        assert np.isfinite(_np(out)).all()

    def test_the_progressive_form_yields_one_dict_per_step(self) -> None:
        model = _tiny_dit()
        gd = GaussianDiffusion.from_name(
            "squaredcos_cap_v2", ORIGINAL_STEPS, timestep_respacing=3
        )
        labels = keras.ops.convert_to_tensor(np.array([0, 1], dtype="int32"))
        outs = list(
            gd.p_sample_loop_progressive(
                self._plain_callable(model, labels),
                shape=(2, 4, 4, 4),
                seed=5,
            )
        )
        assert len(outs) == 3
        for entry in outs:
            assert set(entry) == {"sample", "pred_xstart"}
            assert np.isfinite(_np(entry["sample"])).all()


class TestClassifierFreeGuidanceEndToEnd:
    """``forward_with_cfg`` is handed straight to the sampler, exactly as
    upstream's ``sample.py`` does: duplicated batch, null label ``num_classes``."""

    def test_a_guided_loop_runs(self) -> None:
        model = _tiny_dit()
        gd = GaussianDiffusion.from_name(
            "squaredcos_cap_v2", ORIGINAL_STEPS, timestep_respacing=2
        )
        n = 2
        class_labels = np.array([0, 3], dtype="int32")
        y_null = np.full((n,), TINY_DIT["num_classes"], dtype="int32")
        y = keras.ops.convert_to_tensor(np.concatenate([class_labels, y_null]))

        rng = np.random.default_rng(17)
        z = rng.normal(size=(n, 4, 4, 4)).astype("float32")
        z = keras.ops.convert_to_tensor(np.concatenate([z, z], axis=0))

        def guided(x: Any, t: Any, **kwargs: Any) -> Any:
            return model.forward_with_cfg(x, t, training=False, **kwargs)

        samples = gd.p_sample_loop(
            guided,
            noise=z,
            model_kwargs={"y": y, "cfg_scale": 4.0},
            seed=3,
        )
        assert tuple(keras.ops.shape(samples)) == (2 * n, 4, 4, 4)
        assert np.isfinite(_np(samples)).all()

        # sample.py: `samples, _ = samples.chunk(2, dim=0)`.
        kept = _np(samples)[:n]
        assert kept.shape == (n, 4, 4, 4)
        assert np.isfinite(kept).all()

    def test_the_null_row_is_index_num_classes(self) -> None:
        """A label of ``num_classes`` must be a legal lookup -- the null row
        exists only because ``class_dropout_rate > 0``."""
        model = _tiny_dit()
        y = keras.ops.convert_to_tensor(
            np.array([0, TINY_DIT["num_classes"]], dtype="int32")
        )
        x = keras.ops.convert_to_tensor(
            np.zeros((2, 4, 4, 4), dtype="float32")
        )
        t = keras.ops.zeros((2,), dtype="int32")
        out = model([x, t, y], training=False)
        assert np.isfinite(_np(out)).all()


# ---------------------------------------------------------------------
# Seeding, dtypes, tracing regime, validation
# ---------------------------------------------------------------------


class TestSeedingIsExplicit:
    """Reproducibility comes from the ``seed`` argument, and only from it."""

    @staticmethod
    def _run(seed: Any) -> np.ndarray:
        sched = DDPMSchedule.from_name("squaredcos_cap_v2", 4)
        gd = GaussianDiffusion(sched)
        rng = np.random.default_rng(5)
        model = ConstantModel(
            keras.ops.convert_to_tensor(rng.normal(size=(2, 4, 4, 8)) * 0.3)
        )
        return _np(gd.p_sample_loop(model, shape=(2, 4, 4, 4), seed=seed))

    def test_the_same_seed_reproduces_bit_for_bit(self) -> None:
        np.testing.assert_allclose(
            self._run(1234), self._run(1234), atol=0.0, rtol=0.0
        )

    def test_a_different_seed_gives_a_different_sample(self) -> None:
        assert not np.allclose(self._run(1234), self._run(4321))

    def test_set_random_seed_alone_does_not_reproduce(self) -> None:
        """The reason ``seed`` exists.

        MEASURED on this Keras: ``keras.utils.set_random_seed`` does not re-seed
        an already-created global ``SeedGenerator``, so two runs seeded that way
        DIFFER. A reproducibility check written against ``set_random_seed``
        alone would therefore be measuring nothing. If a future Keras fixes
        this, the first assertion reddens and a reader is told why the explicit
        ``seed`` argument exists -- which is the right outcome, not a silent
        loosening.
        """
        keras.utils.set_random_seed(7)
        first = self._run(None)
        keras.utils.set_random_seed(7)
        second = self._run(None)
        assert not np.allclose(first, second), (
            "keras.utils.set_random_seed now reproduces the global "
            "SeedGenerator's draws; re-read the `seed` plumbing in "
            "diffusion.py before relaxing anything"
        )

        # The explicit path DOES reproduce, in the same process.
        np.testing.assert_allclose(
            self._run(7), self._run(7), atol=0.0, rtol=0.0
        )

    def test_a_seed_generator_can_be_passed_in(self) -> None:
        sched = DDPMSchedule.from_name("squaredcos_cap_v2", 3)
        gd = GaussianDiffusion(sched)
        model = ConstantModel(keras.ops.zeros((2, 4, 4, 8)))
        gen = keras.random.SeedGenerator(seed=11)
        out = gd.p_sample_loop(model, shape=(2, 4, 4, 4), seed=gen)
        assert np.isfinite(_np(out)).all()


class TestDtypeArms:
    """``float32`` and ``float64`` both run, and agree to float32 precision."""

    @pytest.mark.parametrize("dtype", ["float32", "float64"])
    def test_p_mean_variance_preserves_the_input_dtype(self, dtype: str) -> None:
        sched, x, t, model_output = make_case(seed=61)
        gd = GaussianDiffusion(sched)
        xt = keras.ops.cast(keras.ops.convert_to_tensor(x), dtype)
        out = gd.p_mean_variance(
            ConstantModel(
                keras.ops.cast(
                    keras.ops.convert_to_tensor(model_output), dtype
                )
            ),
            xt,
            keras.ops.convert_to_tensor(t),
        )
        for key, value in out.items():
            assert keras.backend.standardize_dtype(value.dtype) == dtype, key
            assert np.isfinite(_np(value)).all(), key

    def test_float64_matches_the_oracle_more_tightly_than_float32(self) -> None:
        sched, x, t, model_output = make_case(seed=62)
        gd = GaussianDiffusion(sched)
        want = oracle_p_mean_variance(
            sched, model_output, x, t, "learned_range", DEFAULT_CLIP_DENOISED
        )
        errors = {}
        for dtype in ("float32", "float64"):
            got = gd.p_mean_variance(
                ConstantModel(
                    keras.ops.cast(
                        keras.ops.convert_to_tensor(model_output), dtype
                    )
                ),
                keras.ops.cast(keras.ops.convert_to_tensor(x), dtype),
                keras.ops.convert_to_tensor(t),
            )
            errors[dtype] = np.abs(_np(got["mean"]) - want["mean"]).max()
        assert errors["float64"] <= errors["float32"]
        assert errors["float64"] < 1e-12

    def test_a_float32_loop_stays_float32(self) -> None:
        sched = DDPMSchedule.from_name("squaredcos_cap_v2", 3)
        gd = GaussianDiffusion(sched)
        model = ConstantModel(keras.ops.zeros((2, 4, 4, 8), dtype="float32"))
        out = gd.p_sample_loop(model, shape=(2, 4, 4, 4), seed=1)
        assert keras.backend.standardize_dtype(out.dtype) == "float32"


class TestTheLoopIsAnEagerPythonLoop:
    """Documents the tracing regime the module docstring states.

    The loop makes exactly one model call per step in Python, so wrapping it in
    ``tf.function`` would unroll ``num_timesteps`` copies of the model graph.
    This arm pins the call count -- the thing that would explode under tracing --
    rather than asserting anything about a tracer.
    """

    def test_one_model_call_per_step(self) -> None:
        for steps in (2, 3, 5):
            sched = DDPMSchedule.from_name("squaredcos_cap_v2", steps)
            gd = GaussianDiffusion(sched)
            model = ConstantModel(keras.ops.zeros((2, 4, 4, 8)))
            gd.p_sample_loop(model, shape=(2, 4, 4, 4), seed=0)
            assert model.calls == steps

    def test_a_single_step_is_ordinary_tensor_code(self) -> None:
        """The per-step methods carry no Python loop, so they trace fine."""
        sched = DDPMSchedule.from_name("squaredcos_cap_v2", 5)
        gd = GaussianDiffusion(sched)
        model = ConstantModel(keras.ops.zeros((2, 4, 4, 8)))
        x = keras.ops.zeros((2, 4, 4, 4))
        t = keras.ops.full((2,), 2, dtype="int32")
        out = gd.p_sample(model, x, t, seed=0)
        assert np.isfinite(_np(out["sample"])).all()


class TestTheClipDenoisedDefaultIsFalseForLatents:
    """D-017: the default DIVERGES from upstream's, matching upstream's own DiT
    call site (``sample.py`` passes ``clip_denoised=False``)."""

    def test_the_constant(self) -> None:
        assert DEFAULT_CLIP_DENOISED is False

    def test_a_default_call_does_not_clip(self) -> None:
        sched, x, t, model_output = make_case(seed=71)
        gd = GaussianDiffusion(sched)
        out = gd.p_mean_variance(
            ConstantModel(keras.ops.convert_to_tensor(model_output)),
            keras.ops.convert_to_tensor(x),
            keras.ops.convert_to_tensor(t),
        )
        pred = _np(out["pred_xstart"])
        assert np.abs(pred).max() > 1.0, (
            "the default is clipping: a latent has been destroyed"
        )

    def test_every_sampler_entry_point_defaults_the_same_way(self) -> None:
        import inspect

        for name in (
            "p_mean_variance",
            "p_sample",
            "p_sample_loop",
            "p_sample_loop_progressive",
            "ddim_sample",
            "ddim_sample_loop",
            "ddim_sample_loop_progressive",
        ):
            signature = inspect.signature(getattr(GaussianDiffusion, name))
            default = signature.parameters["clip_denoised"].default
            assert default is DEFAULT_CLIP_DENOISED, name


class TestTheVarianceInterpolationAgreesWithTheLoss:
    """D-016 lockstep guard for the DUPLICATED ``LEARNED_RANGE`` interpolation.

    ``losses/ddpm_hybrid_loss.py`` computes the same five lines inline. This arm
    rebuilds the loss's whole objective out of :class:`GaussianDiffusion`'s
    ``p_mean_variance`` and the loss module's own KL / decoder-NLL helpers, and
    asserts it equals what the loss returns. Editing either copy of the
    interpolation alone reddens this test, which is what makes the duplication a
    checked invariant rather than a hand-maintained one.
    """

    @pytest.mark.parametrize("schedule_name", ["linear", "squaredcos_cap_v2"])
    def test_the_two_copies_agree(self, schedule_name: str) -> None:
        steps, channels = 50, 4
        rng = np.random.default_rng(81)
        x_start = rng.normal(size=(3, 4, 4, channels))
        noise = rng.normal(size=x_start.shape)
        t = np.array([0, 7, 49], dtype=np.int32)
        y_pred = rng.normal(size=(3, 4, 4, 2 * channels)) * 0.5

        loss = DDPMHybridLoss(
            schedule_name=schedule_name,
            num_timesteps=steps,
            in_channels=channels,
        )
        gd = GaussianDiffusion(
            DDPMSchedule.from_name(schedule_name, steps),
            model_var_type="learned_range",
        )

        t_plane = np.broadcast_to(
            t.reshape(-1, 1, 1, 1).astype(np.float64), x_start.shape[:-1] + (1,)
        )
        y_true = np.concatenate([noise, x_start, t_plane], axis=-1)

        got = _np(
            loss.call(
                keras.ops.convert_to_tensor(y_true),
                keras.ops.convert_to_tensor(y_pred),
            )
        )

        # Rebuild the same objective through the SAMPLER's copy.
        x_t = gd.q_sample(
            keras.ops.convert_to_tensor(x_start),
            keras.ops.convert_to_tensor(t),
            noise=keras.ops.convert_to_tensor(noise),
        )
        out = gd.p_mean_variance(
            ConstantModel(keras.ops.convert_to_tensor(y_pred)),
            x_t,
            keras.ops.convert_to_tensor(t),
            clip_denoised=False,
        )
        true_mean, _, true_log_var = gd.q_posterior_mean_variance(
            keras.ops.convert_to_tensor(x_start),
            x_t,
            keras.ops.convert_to_tensor(t),
        )
        kl = _mean_flat(
            _normal_kl(
                true_mean, true_log_var, out["mean"], out["log_variance"]
            )
        ) / np.log(2.0)
        nll = _mean_flat(
            -_discretized_gaussian_log_likelihood(
                keras.ops.convert_to_tensor(x_start),
                means=out["mean"],
                log_scales=0.5 * out["log_variance"],
            )
        ) / np.log(2.0)
        vb = np.where(t == 0, _np(nll), _np(kl))
        mse = np.mean(
            (noise - y_pred[..., :channels]) ** 2, axis=(1, 2, 3)
        )
        np.testing.assert_allclose(got, mse + vb, atol=1e-6, rtol=0.0)


class TestConstructionAndValidation:
    """Factories, the config bridge, and the named errors."""

    def test_from_config_learned_range(self) -> None:
        config = DiffusionConfig(
            input_size=4,
            in_channels=4,
            num_classes=5,
            num_timesteps=ORIGINAL_STEPS,
            schedule_name="squaredcos_cap_v2",
            learn_sigma=True,
        )
        gd = GaussianDiffusion.from_config(config)
        assert gd.model_var_type == "learned_range"
        assert gd.num_timesteps == ORIGINAL_STEPS
        assert gd.timestep_map is None

    def test_from_config_without_learn_sigma_uses_a_fixed_variance(self) -> None:
        config = DiffusionConfig(
            input_size=4,
            in_channels=4,
            num_classes=5,
            num_timesteps=ORIGINAL_STEPS,
            schedule_name="squaredcos_cap_v2",
            learn_sigma=False,
        )
        gd = GaussianDiffusion.from_config(config, timestep_respacing=4)
        assert gd.model_var_type == "fixed_small"
        assert gd.num_timesteps == 4
        model = RecordingModel(out_channels=4)
        out = gd.p_mean_variance(
            model,
            keras.ops.zeros((2, 4, 4, 4)),
            keras.ops.full((2,), 3, dtype="int32"),
        )
        assert np.isfinite(_np(out["log_variance"])).all()

    def test_ddim_style_respacing_string(self) -> None:
        gd = GaussianDiffusion.from_name(
            "squaredcos_cap_v2", 20, timestep_respacing="ddim5"
        )
        assert gd.num_timesteps == 5
        assert list(gd.timestep_map) == [0, 4, 8, 12, 16]

    def test_an_empty_respacing_keeps_the_full_chain(self) -> None:
        for spec in (None, ""):
            gd = GaussianDiffusion.from_name("squaredcos_cap_v2", 6, spec)
            assert gd.num_timesteps == 6
            assert gd.timestep_map is None

    def test_a_bad_var_type_raises(self) -> None:
        sched = DDPMSchedule.from_name("squaredcos_cap_v2", 4)
        with pytest.raises(ValueError, match="model_var_type"):
            GaussianDiffusion(sched, model_var_type="learned_wrong")

    def test_a_bad_mean_type_raises(self) -> None:
        sched = DDPMSchedule.from_name("squaredcos_cap_v2", 4)
        with pytest.raises(ValueError, match="model_mean_type"):
            GaussianDiffusion(sched, model_mean_type="x_prev")

    def test_a_mismatched_timestep_map_raises(self) -> None:
        sched = DDPMSchedule.from_name("squaredcos_cap_v2", 4)
        with pytest.raises(ValueError, match="timestep_map"):
            GaussianDiffusion(sched, timestep_map=np.array([0, 1]))

    def test_a_learn_sigma_false_model_under_learned_range_raises(self) -> None:
        sched = DDPMSchedule.from_name("squaredcos_cap_v2", 4)
        gd = GaussianDiffusion(sched, model_var_type="learned_range")
        with pytest.raises(ValueError, match="learn_sigma=False"):
            gd.p_mean_variance(
                ConstantModel(keras.ops.zeros((2, 4, 4, 4))),
                keras.ops.zeros((2, 4, 4, 4)),
                keras.ops.zeros((2,), dtype="int32"),
            )

    def test_a_loop_without_shape_or_noise_raises(self) -> None:
        sched = DDPMSchedule.from_name("squaredcos_cap_v2", 4)
        gd = GaussianDiffusion(sched)
        with pytest.raises(ValueError, match="shape"):
            gd.p_sample_loop(ConstantModel(keras.ops.zeros((2, 4, 4, 8))))

    def test_it_is_not_a_keras_object(self) -> None:
        """Deliberately unregistered: it holds NumPy tables and no weights, the
        same call as :class:`DDPMSchedule`."""
        sched = DDPMSchedule.from_name("squaredcos_cap_v2", 4)
        gd = GaussianDiffusion(sched)
        assert not isinstance(gd, keras.layers.Layer)
        assert not hasattr(gd, "get_config")


# ---------------------------------------------------------------------
