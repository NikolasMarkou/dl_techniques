"""DDPM hybrid training objective (epsilon MSE + frozen-out variational bound).

This module provides :class:`DDPMHybridLoss`, the objective a class-conditional
latent Diffusion Transformer is trained under: the ``LossType.MSE`` +
``ModelVarType.LEARNED_RANGE`` branch of the upstream ``training_losses``
(``reference/diffusion/gaussian_diffusion.py:463-494``, ported here as part of
plan ``plan-2026-09-02T170923-1285ed83``). It is the *full* published objective,
not the epsilon-MSE half of it: the model emits ``2 * C`` channels, the second
half of which are variance-interpolation logits that plain MSE cannot train at
all, and the variational-bound term is the only thing that supervises them.

Why the target tensor is packed
-------------------------------
A ``keras.losses.Loss`` is handed exactly ``(y_true, y_pred, sample_weight)``.
The variational-bound term additionally needs the clean latent ``x_start`` and
the per-sample diffusion timestep ``t``, and neither is derivable from an
epsilon target. ``sample_weight`` is NOT a free side channel -- Keras multiplies
the per-sample loss by it, so smuggling ``t`` through it would silently corrupt
the objective. The remaining honest option, and the one taken, is to pack every
extra quantity into ``y_true`` along the channel axis. That keeps the training
loop on stock ``compile()`` / ``fit()`` with no ``train_step`` override.

The consequence is a ``y_true`` whose channel count (``2C + 1``) deliberately
differs from ``y_pred``'s (``2C``). That is the contract, not a bug.

.. code-block:: text

    y_true  [B, H, W, 2C+1]                 y_pred  [B, H, W, 2C]
    ┌───────────────┬───────────────┬───┐   ┌───────────────┬───────────────┐
    │ noise  [0:C]  │ x_start [C:2C]│ t │   │ eps_pred[0:C] │ var_logits    │
    │     epsilon   │  clean latent │pln│   │               │      [C:2C]   │
    └───────┬───────┴───────┬───────┴─┬─┘   └───────┬───────┴───────┬───────┘
            │               │         │             │               │
            │               │         │             │        ┌──────▼──────┐
            │               │         │             │        │ stop_gradient
            │               │         │             ▼        │  on eps only │
            │               │         │        ┌─────────┐   └──────┬──────┘
            │               │         │        │  MSE    │          │
            │               │         │        │ term    │          │
            │               │         │        └────┬────┘          │
            ▼               ▼         ▼             │               │
    ┌───────────────────────────────────┐           │               │
    │ x_t = sqrt_ac[t] * x_start        │           │               │
    │      ⊕ sqrt_1mac[t] * noise       │───────────┼───────────────┤
    │   (RE-DERIVED here, never passed) │           │               │
    └───────────────┬───────────────────┘           │               │
                    │                               │               │
                    ▼                               │               ▼
    ┌───────────────────────────────────┐           │   ┌───────────────────┐
    │ q(x_{t-1} | x_t, x_0):            │           │   │ p_mean_variance:  │
    │   true_mean, true_log_var         │──────────────▶│  frac=(v+1)/2     │
    └───────────────────────────────────┘           │   │  log var = frac   │
                                                    │   │   * log(beta[t])  │
                                                    │   │   ⊕ (1-frac)      │
                                                    │   │   * min_log[t]    │
                                                    │   └─────────┬─────────┘
                                                    │             │
                                                    │             ▼
                                                    │   ┌───────────────────┐
                                                    │   │ normal_kl / log 2 │
                                                    │   │ (decoder NLL at   │
                                                    │   │  t == 0)   = vb   │
                                                    │   └─────────┬─────────┘
                                                    │             │
                                                    └──────⊕──────┘
                                                           │
                                                           ▼
                                                   loss  [B]  per-sample

Three properties that are load-bearing
--------------------------------------
1.  **The variational bound is frozen out of the mean prediction.** Upstream
    builds ``frozen_out = cat([model_output.detach(), model_var_values])`` and
    feeds *that* to the bound, so the bound trains the variance channels and
    only the variance channels. Removing the ``stop_gradient`` leaves every
    shape, every finiteness check and the loss VALUE itself unchanged -- only
    the gradient w.r.t. the epsilon channels moves. It is pinned by
    ``tests/test_losses/test_ddpm_hybrid_loss.py``.
2.  **``x_t`` is re-derived from the schedule, never handed in.** The bound
    needs the posterior tables anyway, so carrying ``x_t`` would double the
    target tensor while only halving the pipeline/loss coupling. Both sides
    instead build a :class:`~dl_techniques.utils.ddpm_schedule.DDPMSchedule`
    from the same configuration.
3.  **``call()`` returns one value per sample, shape ``[B]``.** A Keras loss
    whose ``call()`` returns a scalar does not "ignore" ``sample_weight``, it
    CORRUPTS it: the scalar broadcasts and every row is charged the batch
    aggregate. See ``tests/test_losses/test_the_premature_scalar_family_is_pinned.py``.

The channel axis is ``-1`` throughout. Upstream is NCHW and splits on
``dim=1``; every split, concatenation and ``mean_flat`` here was re-derived for
channels-last rather than transcribed.

References:
    - Ho et al., 2020. Denoising Diffusion Probabilistic Models.
      (https://arxiv.org/abs/2006.11239) -- the epsilon-prediction MSE term and
      the fixed-variance reverse process.
    - Nichol & Dhariwal, 2021. Improved Denoising Diffusion Probabilistic
      Models. (https://arxiv.org/abs/2102.09672) -- the hybrid objective
      ``L_simple + lambda * L_vlb`` and the ``LEARNED_RANGE`` parameterization
      that interpolates ``log`` variance between the posterior and ``beta_t``.
    - Peebles & Xie, 2022. Scalable Diffusion Models with Transformers.
      (https://arxiv.org/abs/2212.09748) -- the consumer this objective exists
      for; DiT trains with ``learn_sigma=True`` under exactly this loss.
"""

import math
from typing import Any, Dict, Sequence, Tuple

import keras
import numpy as np

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.keras_registration import register_dl_technique
from dl_techniques.utils.ddpm_schedule import DDPMSchedule

# ---------------------------------------------------------------------

__all__ = ["DDPMHybridLoss"]

# Natural log of 2. The variational bound is reported in BITS, not nats, so the
# KL and the decoder NLL are both divided by this.
_LOG_2 = math.log(2.0)


# ---------------------------------------------------------------------
# Backend-agnostic helpers (transcribed from reference/diffusion/diffusion_utils.py)
# ---------------------------------------------------------------------


def _mean_flat(tensor: Any) -> Any:
    """Mean over every non-batch axis.

    Upstream's ``mean_flat`` (``gaussian_diffusion.py:12-14``). For a
    ``[B, H, W, C]`` tensor this is a mean over ``(1, 2, 3)``.

    :param tensor: Tensor of rank >= 1.
    :type tensor: Any
    :return: Tensor of shape ``[B]``.
    :rtype: Any
    """
    rank = len(keras.ops.shape(tensor))
    if rank <= 1:
        return tensor
    return keras.ops.mean(tensor, axis=tuple(range(1, rank)))


def _normal_kl(mean1: Any, logvar1: Any, mean2: Any, logvar2: Any) -> Any:
    """KL divergence between two diagonal Gaussians, elementwise.

    Transcribed from ``diffusion_utils.normal_kl``.

    :param mean1: Mean of the first (reference) Gaussian.
    :type mean1: Any
    :param logvar1: Log-variance of the first Gaussian.
    :type logvar1: Any
    :param mean2: Mean of the second Gaussian.
    :type mean2: Any
    :param logvar2: Log-variance of the second Gaussian.
    :type logvar2: Any
    :return: Elementwise ``KL(N1 || N2)``, same shape as the broadcast inputs.
    :rtype: Any
    """
    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + keras.ops.exp(logvar1 - logvar2)
        + keras.ops.square(mean1 - mean2) * keras.ops.exp(-logvar2)
    )


def _approx_standard_normal_cdf(x: Any) -> Any:
    """Tanh approximation of the standard-normal CDF.

    Transcribed from ``diffusion_utils.approx_standard_normal_cdf``.

    :param x: Input tensor.
    :type x: Any
    :return: Approximate ``Phi(x)``, same shape as ``x``.
    :rtype: Any
    """
    coeff = math.sqrt(2.0 / math.pi)
    return 0.5 * (
        1.0
        + keras.ops.tanh(coeff * (x + 0.044715 * keras.ops.power(x, 3)))
    )


def _discretized_gaussian_log_likelihood(
    x: Any,
    means: Any,
    log_scales: Any,
) -> Any:
    """Log-likelihood of a Gaussian discretized onto a 1/255 grid.

    Transcribed from ``diffusion_utils.discretized_gaussian_log_likelihood``.
    This is the decoder term of the variational bound, used at ``t == 0`` in
    place of the KL. It assumes ``x`` was uint8 data rescaled to ``[-1, 1]``;
    upstream applies it unchanged to VAE latents, which are NOT in that range,
    and this port reproduces that as-is rather than "fixing" it.

    :param x: Target tensor.
    :type x: Any
    :param means: Predicted means, same shape as ``x``.
    :type means: Any
    :param log_scales: Predicted log standard deviations, same shape as ``x``.
    :type log_scales: Any
    :return: Elementwise log-probabilities, same shape as ``x``.
    :rtype: Any
    """
    centered_x = x - means
    inv_stdv = keras.ops.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = _approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = _approx_standard_normal_cdf(min_in)
    log_cdf_plus = keras.ops.log(keras.ops.maximum(cdf_plus, 1e-12))
    log_one_minus_cdf_min = keras.ops.log(
        keras.ops.maximum(1.0 - cdf_min, 1e-12)
    )
    cdf_delta = cdf_plus - cdf_min
    return keras.ops.where(
        x < -0.999,
        log_cdf_plus,
        keras.ops.where(
            x > 0.999,
            log_one_minus_cdf_min,
            keras.ops.log(keras.ops.maximum(cdf_delta, 1e-12)),
        ),
    )


# ---------------------------------------------------------------------


@register_dl_technique(package="dl_techniques.losses.ddpm_hybrid_loss")
class DDPMHybridLoss(keras.losses.Loss):
    """Epsilon MSE plus a frozen-out variational bound, for LEARNED_RANGE DDPMs.

    Reproduces upstream's ``LossType.MSE`` + ``ModelVarType.LEARNED_RANGE``
    training objective under stock ``compile()`` / ``fit()``:
    ``loss = mean_flat((noise - eps_pred) ** 2) + vb``, where ``vb`` is the
    variational bound in bits, evaluated against a model output whose epsilon
    half has been detached.

    .. code-block:: text

        ┌──────────────────────────── y_true [B, H, W, 2C+1] ───────────────┐
        │  [0:C] noise (epsilon)  │ [C:2C] x_start │ [2C:2C+1] t as a plane │
        └───────────┬─────────────┴────────┬───────┴──────────┬─────────────┘
                    │                      │                  │
                    └──────────► q_sample ◄┘◄─────────────────┘
                                    │
                                    ▼  x_t [B, H, W, C]
        ┌──────────────────────────── y_pred [B, H, W, 2C] ────────────────┐
        │      [0:C] eps_pred          │     [C:2C] variance logits        │
        └───────────┬──────────────────┴──────────────┬────────────────────┘
                    │                                 │
             MSE vs noise                   stop_gradient(eps_pred)
                    │                                 │
                    │                        variational bound (bits)
                    │                                 │
                    └────────────────⊕────────────────┘
                                     ▼
                              loss [B] per sample

    :param schedule_name: Beta-schedule name handed to
        :meth:`~dl_techniques.utils.ddpm_schedule.DDPMSchedule.from_name`;
        ``'linear'`` or ``'squaredcos_cap_v2'``. ``'linear'`` is NOT defined at
        every chain length: its ``beta_end = 20 / num_timesteps`` leaves ``(0, 1]``
        below ``T = 20``, and the measured accepted set is ``{1}`` union
        ``[20, inf)`` -- ``np.linspace(a, b, 1)`` drops the illegal endpoint. It
        is not a floor and no threshold expression states it, so the schedule is
        BUILT and its own ValueError is the arbiter (D-010).
    :type schedule_name: str
    :param num_timesteps: Length of the diffusion chain, ``T``.
    :type num_timesteps: int
    :param in_channels: Number of latent channels ``C``. ``y_pred`` must carry
        ``2 * C`` channels and ``y_true`` must carry ``2 * C + 1``.
    :type in_channels: int
    :param name: Loss name. Defaults to ``'ddpm_hybrid_loss'``.
    :type name: str
    :param reduction: Reduction over the batch axis. Defaults to
        ``'sum_over_batch_size'``, which reproduces upstream's ``loss.mean()``.
    :type reduction: str
    :param kwargs: Forwarded to :class:`keras.losses.Loss`.
    :type kwargs: Any
    :raises ValueError: If ``in_channels`` is not positive.

    Example:

    .. code-block:: python

        import keras
        from dl_techniques.losses import DDPMHybridLoss

        model.compile(
            optimizer=keras.optimizers.AdamW(learning_rate=1e-4),
            loss=DDPMHybridLoss(
                schedule_name="linear", num_timesteps=1000, in_channels=4
            ),
        )
    """

    def __init__(
        self,
        schedule_name: str = "linear",
        num_timesteps: int = 1000,
        in_channels: int = 4,
        name: str = "ddpm_hybrid_loss",
        reduction: str = "sum_over_batch_size",
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, reduction=reduction, **kwargs)

        if in_channels <= 0:
            raise ValueError(
                f"in_channels must be positive, got {in_channels}"
            )

        self.schedule_name = schedule_name
        self.num_timesteps = num_timesteps
        self.in_channels = in_channels

        # The schedule validates `schedule_name` and `num_timesteps` for us and
        # raises a named ValueError on a bad pair.
        self.schedule: DDPMSchedule = DDPMSchedule.from_name(
            schedule_name=schedule_name, num_timesteps=num_timesteps
        )

        # Every table stays a float64 NUMPY array on the instance and is
        # converted inside `call()`. A constant materialized with
        # `ops.convert_to_tensor` in `__init__` binds to whichever FuncGraph
        # traced it, and `fit()` then dies with `InaccessibleTensorError`.
        self._log_betas: np.ndarray = np.log(self.schedule.betas)

        logger.info(
            "DDPMHybridLoss: schedule=%s, T=%d, C=%d (epsilon MSE + "
            "frozen-out variational bound over the variance channels)",
            schedule_name,
            num_timesteps,
            in_channels,
        )

    # -----------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------

    def _gather(
        self,
        table: np.ndarray,
        t: Any,
        n_broadcast_axes: int,
        dtype: str,
    ) -> Any:
        """Gather one table entry per sample and shape it for broadcasting.

        Upstream's ``_extract_into_tensor`` (``gaussian_diffusion.py:545-550``).

        :param table: 1-D float64 NumPy table of length ``T``.
        :type table: np.ndarray
        :param t: Integer timestep tensor of shape ``[B]``.
        :type t: Any
        :param n_broadcast_axes: Number of trailing singleton axes to append.
        :type n_broadcast_axes: int
        :param dtype: Floating dtype the result is produced in.
        :type dtype: str
        :return: Tensor of shape ``[B] + [1] * n_broadcast_axes``.
        :rtype: Any
        """
        values = keras.ops.take(
            keras.ops.cast(keras.ops.convert_to_tensor(table), dtype),
            t,
            axis=0,
        )
        return keras.ops.reshape(
            values, (-1,) + (1,) * n_broadcast_axes
        )

    @staticmethod
    def _compute_dtype(y_pred: Any) -> str:
        """Floating dtype the objective is evaluated in.

        Half precision is promoted to ``float32``: the bound exponentiates a
        log-variance and divides by it, which underflows in ``float16`` long
        before any shape or finiteness check would notice. ``float64`` inputs
        are honoured so the test oracle can compare in full precision.

        :param y_pred: The model output.
        :type y_pred: Any
        :return: ``'float32'`` or ``'float64'``.
        :rtype: str
        """
        # DECISION plan-2026-09-02T170923-1285ed83/D-018
        # `getattr(dtype, "name", None) or str(dtype)` rather than
        # `keras.backend.standardize_dtype`. WHAT NOT TO DO: do not "simplify"
        # this back to the standardize_dtype call. It is a Keras-2 residue that
        # `tests/test_the_keras2_backend_calls_are_gone.py` bans across the whole
        # of `src/` -- `losses/` included, so a regression here IS caught now.
        # (It used to be a `models/`-only sweep that could not see this file;
        # that scope, and the class that held it, are both gone.) The sibling
        # `dit/diffusion.py:449-460` spells it the same way. `str`
        # alone is not enough either: a `tf.DType` stringifies as
        # "<dtype: 'float64'>". Same call as `bit_diffusion/sde.py:123`.
        # See decisions.md D-018.
        raw = y_pred.dtype
        name = getattr(raw, "name", None) or str(raw)
        return "float64" if name == "float64" else "float32"

    def _unpack(
        self,
        y_true: Any,
        y_pred: Any,
        dtype: str,
    ) -> Tuple[Any, Any, Any, Any, Any]:
        """Split the packed target and the model output into their five parts.

        :param y_true: Packed target, ``[B, ..., 2C+1]``.
        :type y_true: Any
        :param y_pred: Model output, ``[B, ..., 2C]``.
        :type y_pred: Any
        :param dtype: Floating dtype to cast the float parts to.
        :type dtype: str
        :return: ``(noise, x_start, t, eps_pred, var_logits)`` where ``t`` has
            shape ``[B]`` and integer dtype.
        :rtype: Tuple[Any, Any, Any, Any, Any]
        """
        c = self.in_channels

        # DECISION plan-2026-09-02T170923-1285ed83/D-002
        # This channel layout is a hand-maintained contract between the data
        # pipeline (src/train/dit/synthetic_data.py) and this loss, and NOTHING
        # about it has a shape symptom: swapping the [0:C] and [C:2C] halves
        # yields a target of the identical shape and dtype that trains a
        # plausible, wrong model. Do NOT "simplify" it by making y_true a plain
        # epsilon target -- the variance logits in y_pred[..., C:2C] would then
        # have no supervision at all -- and do NOT move `t` onto sample_weight,
        # which Keras MULTIPLIES the per-sample loss by. decisions.md D-002.
        noise = keras.ops.cast(y_true[..., 0:c], dtype)
        x_start = keras.ops.cast(y_true[..., c: 2 * c], dtype)
        t_plane = y_true[..., 2 * c]
        eps_pred = keras.ops.cast(y_pred[..., 0:c], dtype)
        var_logits = keras.ops.cast(y_pred[..., c: 2 * c], dtype)

        # `t` is broadcast over every spatial position; any one of them is the
        # timestep. Read the first, and round before casting so a float32 plane
        # cannot truncate an exact integer downwards.
        t_flat = keras.ops.reshape(t_plane, (keras.ops.shape(t_plane)[0], -1))
        t = keras.ops.cast(
            keras.ops.round(keras.ops.cast(t_flat[:, 0], "float32")), "int32"
        )

        return noise, x_start, t, eps_pred, var_logits

    def _validate_static_shapes(self, y_true: Any, y_pred: Any) -> None:
        """Reject a mis-packed target as early as the static shape allows.

        :param y_true: Packed target.
        :type y_true: Any
        :param y_pred: Model output.
        :type y_pred: Any
        :raises ValueError: If a statically known channel count disagrees with
            ``in_channels``.
        """
        c = self.in_channels
        true_shape: Sequence[Any] = getattr(y_true, "shape", ())
        pred_shape: Sequence[Any] = getattr(y_pred, "shape", ())

        if len(true_shape) > 0 and true_shape[-1] is not None:
            if int(true_shape[-1]) != 2 * c + 1:
                raise ValueError(
                    f"DDPMHybridLoss expects y_true with {2 * c + 1} channels "
                    f"([0:{c}]=noise, [{c}:{2 * c}]=x_start, "
                    f"[{2 * c}:{2 * c + 1}]=t) for in_channels={c}, got "
                    f"{int(true_shape[-1])}."
                )
        if len(pred_shape) > 0 and pred_shape[-1] is not None:
            if int(pred_shape[-1]) != 2 * c:
                raise ValueError(
                    f"DDPMHybridLoss expects y_pred with {2 * c} channels "
                    f"([0:{c}]=epsilon, [{c}:{2 * c}]=variance logits) for "
                    f"in_channels={c}, got {int(pred_shape[-1])}."
                )

    # -----------------------------------------------------------------
    # Loss
    # -----------------------------------------------------------------

    def call(self, y_true: Any, y_pred: Any) -> Any:
        """Compute the per-sample hybrid loss.

        :param y_true: Packed target ``[B, H, W, 2C+1]`` -- see the module
            docstring for the channel layout.
        :type y_true: Any
        :param y_pred: Model output ``[B, H, W, 2C]``.
        :type y_pred: Any
        :return: Per-sample loss of shape ``[B]``. NOT a scalar: the batch
            reduction is Keras' job, and a scalar here would corrupt
            ``sample_weight``.
        :rtype: Any
        :raises ValueError: If a statically known channel count is wrong.
        """
        self._validate_static_shapes(y_true, y_pred)

        dtype = self._compute_dtype(y_pred)
        noise, x_start, t, eps_pred, var_logits = self._unpack(
            y_true, y_pred, dtype
        )
        n_bcast = len(keras.ops.shape(x_start)) - 1

        def gather(table: np.ndarray) -> Any:
            return self._gather(table, t, n_bcast, dtype)

        sched = self.schedule

        # --- q_sample: re-derive x_t from (x_start, t, noise) -------------
        x_t = (
            gather(sched.sqrt_alphas_cumprod) * x_start
            + gather(sched.sqrt_one_minus_alphas_cumprod) * noise
        )

        # --- the simple epsilon-prediction term ---------------------------
        mse = _mean_flat(keras.ops.square(noise - eps_pred))

        # --- the variational bound, with the mean prediction frozen -------
        # Upstream: frozen_out = cat([model_output.detach(), model_var_values]).
        # The bound must train the variance channels WITHOUT touching the mean.
        frozen_eps = keras.ops.stop_gradient(eps_pred)

        # p_mean_variance, LEARNED_RANGE branch, clip_denoised=False.
        min_log = gather(sched.posterior_log_variance_clipped)
        max_log = gather(self._log_betas)
        frac = (var_logits + 1.0) / 2.0
        model_log_variance = frac * max_log + (1.0 - frac) * min_log

        pred_xstart = (
            gather(sched.sqrt_recip_alphas_cumprod) * x_t
            - gather(sched.sqrt_recipm1_alphas_cumprod) * frozen_eps
        )
        coef1 = gather(sched.posterior_mean_coef1)
        coef2 = gather(sched.posterior_mean_coef2)
        model_mean = coef1 * pred_xstart + coef2 * x_t

        # q_posterior_mean_variance against the TRUE x_start.
        true_mean = coef1 * x_start + coef2 * x_t
        true_log_variance_clipped = min_log

        kl = _normal_kl(
            true_mean, true_log_variance_clipped, model_mean, model_log_variance
        )
        kl = _mean_flat(kl) / _LOG_2

        decoder_nll = -_discretized_gaussian_log_likelihood(
            x_start, means=model_mean, log_scales=0.5 * model_log_variance
        )
        decoder_nll = _mean_flat(decoder_nll) / _LOG_2

        # At t == 0 there is no q(x_{-1} | ...), so the bound is the decoder NLL.
        vb = keras.ops.where(keras.ops.equal(t, 0), decoder_nll, kl)

        return mse + vb

    # -----------------------------------------------------------------
    # Serialization
    # -----------------------------------------------------------------

    def get_config(self) -> Dict[str, Any]:
        """Return the constructor arguments.

        The schedule PARAMETERS round-trip; the derived float64 tables never
        do. They are a pure function of the parameters, and serializing a
        thousand-entry array would let a checkpoint disagree with the code that
        derives it.

        :return: Serializable configuration.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update(
            {
                "schedule_name": self.schedule_name,
                "num_timesteps": self.num_timesteps,
                "in_channels": self.in_channels,
            }
        )
        return config
