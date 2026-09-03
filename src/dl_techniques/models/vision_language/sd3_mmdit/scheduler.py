"""Rectified-flow Euler scheduler, built by ``FlowMatchEulerScheduler``, for
SD3-style MMDiT training and sampling.

Ports the Stable Diffusion 3 ``FlowMatchEulerDiscreteScheduler``: a straight-
line interpolation between clean data and noise, ``x_t = (1-t)*x0 + t*noise``,
whose velocity target ``dx/dt`` is the constant ``noise - x0``. Reverse
sampling integrates that velocity from ``t=1`` (noise) down to ``t=0`` (data)
with a negative Euler step. This is a plain frozen dataclass, not a Keras
layer, since nothing here is trainable. Its in-graph tensor math
(``add_noise``, ``velocity_target``, ``euler_step``) uses ``keras.ops`` so a
trainer can call it inside a ``tf.function``; its host-side scheduling
(``sample_logit_normal_t``, ``logit_normal_weight``, ``timesteps``) needs the
inverse normal CDF and logistic sigmoid, which ``keras.ops`` has no
backend-agnostic form of, so it runs in NumPy/SciPy in float64 and casts to
float32 on return.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.special import ndtri, expit

from dl_techniques.utils.logger import logger


@dataclass(frozen=True)
class FlowMatchEulerScheduler:
    """Rectified-flow Euler scheduler (forward noising + reverse Euler).

    A frozen dataclass holding the rectified-flow hyperparameters. The
    interpolation is a straight line between clean data (``t=0``) and pure
    noise (``t=1``); the velocity target is the constant ``noise - x0``.

    :param num_train_timesteps: Number of discrete training timesteps; the
        continuous ``t`` is conceptually ``step / num_train_timesteps``.
        Defaults to ``1000``.
    :type num_train_timesteps: int
    :param shift: SD3 static time-shift applied to the logit-normal samples
        and the inference time grid: ``t -> shift * t / (1 + (shift - 1) * t)``.
        ``shift = 1`` disables the warp. Defaults to ``3.0``.
    :type shift: float
    :param logit_mean: Mean of the logit-normal time-sampling distribution
        (pre-shift). Defaults to ``0.0``.
    :type logit_mean: float
    :param logit_std: Standard deviation of the logit-normal time-sampling
        distribution. Defaults to ``1.0``.
    :type logit_std: float
    """

    num_train_timesteps: int = 1000
    shift: float = 3.0
    logit_mean: float = 0.0
    logit_std: float = 1.0

    # ------------------------------------------------------------------ #
    # In-graph tensor math (keras.ops; also valid on NumPy arrays).
    # ------------------------------------------------------------------ #
    def add_noise(self, x0, noise, t):
        """Rectified-flow forward interpolation ``x_t = (1 - t) * x0 + t * noise``.

        :param x0: Clean data tensor, any shape.
        :param noise: Noise tensor broadcastable to ``x0``.
        :param t: Time in ``[0, 1]``, a scalar or a tensor broadcastable to
            ``x0`` (e.g. ``(B, 1, 1, 1)`` for a per-sample time).
        :return: The noised sample ``x_t``, at the broadcast shape of the
            inputs. Equals ``x0`` at ``t=0`` and ``noise`` at ``t=1``.
        """
        return (1.0 - t) * x0 + t * noise

    def velocity_target(self, x0, noise):
        """Rectified-flow velocity target ``noise - x0`` (constant in ``t``).

        The trainer's MSE target and the sampler's Euler integrand both use
        this quantity.

        :param x0: Clean data tensor.
        :param noise: Noise tensor broadcastable to ``x0``.
        :return: The velocity ``noise - x0``, at the broadcast shape of the inputs.
        """
        return noise - x0

    def euler_step(self, x_t, v_pred, t, t_next):
        """One reverse Euler integration step of ``dx/dt = v``.

        Integrates ``x_next = x_t + v_pred * (t_next - t)``. Sampling runs
        from ``t=1`` (noise) down to ``t=0`` (data), so ``t_next < t`` and
        ``dt`` is negative. With the true velocity this single step is exact
        for a straight-line path: stepping ``t=1 -> t=0`` recovers ``x0``.

        :param x_t: Current sample at time ``t``.
        :param v_pred: Predicted (or true) velocity at ``t``, same shape as ``x_t``.
        :param t: Current time, scalar or tensor in ``[0, 1]``.
        :param t_next: Next time, scalar or tensor in ``[0, 1]``, less than
            ``t`` when sampling.
        :return: The integrated sample ``x_next`` at time ``t_next``.
        """
        # DECISION plan_2026-06-12_dfce0712/D-007: dt = t_next - t stays signed,
        # never abs() -- sampling descends t=1->t=0 so dt is negative, which recovers x0 exactly in one step. See decisions.md.
        return x_t + v_pred * (t_next - t)

    # ------------------------------------------------------------------ #
    # Host-side scheduling (NumPy / scipy; float64 internal -> float32 out).
    # ------------------------------------------------------------------ #
    def _apply_shift(self, t: np.ndarray) -> np.ndarray:
        """SD3 static time-shift warp ``t -> shift*t / (1 + (shift-1)*t)``."""
        if self.shift == 1.0:
            return t
        return self.shift * t / (1.0 + (self.shift - 1.0) * t)

    def sample_logit_normal_t(
        self, batch_size: int, seed: int | None = None
    ) -> np.ndarray:
        """Draw SD3 logit-normal training times (with the static shift warp).

        Draws ``u ~ Uniform(0, 1)``, maps it to a logit-normal time via the
        inverse normal CDF / sigmoid, then applies the SD3 static shift::

            t = sigmoid(logit_mean + logit_std * ndtri(u))
            t = shift * t / (1 + (shift - 1) * t)

        :param batch_size: Number of times to draw.
        :param seed: Optional RNG seed for reproducibility.
        :return: A float32 array of shape ``(batch_size,)`` with values in
            the open interval ``(0, 1)``.
        """
        rng = np.random.default_rng(seed)
        # Open interval (0, 1): avoid ndtri(0)=-inf / ndtri(1)=+inf.
        eps = np.finfo(np.float64).tiny
        u = rng.uniform(eps, 1.0 - eps, size=batch_size).astype(np.float64)
        z = ndtri(u)
        t = expit(self.logit_mean + self.logit_std * z)
        t = self._apply_shift(t)
        return t.astype(np.float32)

    def logit_normal_weight(self, t) -> np.ndarray:
        """SD3 Eq.(19) loss weight ``w(t) = 1 / pdf_logitnormal(t)``.

        Reciprocal of the logit-normal probability density, reproducing the
        PyTorch ``logit_normal_weighting``::

            term1 = t * (1 - t) * std * sqrt(2 * pi)
            term2 = exp( (logit(t) - mean)^2 / (2 * std^2) )
            w     = term1 * term2

        where ``logit(t) = log(t / (1 - t))``. ``t`` is clamped to
        ``(eps, 1 - eps)``. This method only computes the weight; the trainer
        multiplies it into the per-sample loss.

        :param t: Time(s) in ``(0, 1)``, a Python float or array-like.
        :return: A float32 array of weights, same shape as ``t``, positive
            and finite away from the boundaries.
        """
        t_arr = np.asarray(t, dtype=np.float64)
        eps = 1e-7
        t_arr = np.clip(t_arr, eps, 1.0 - eps)
        mean = self.logit_mean
        std = self.logit_std
        logit_t = np.log(t_arr / (1.0 - t_arr))
        term1 = t_arr * (1.0 - t_arr) * std * math.sqrt(2.0 * math.pi)
        term2 = np.exp((logit_t - mean) ** 2 / (2.0 * std ** 2))
        w = term1 * term2
        return w.astype(np.float32)

    def timesteps(self, num_inference_steps: int) -> np.ndarray:
        """Descending sampling time grid from close to 1 down to 0.

        Builds ``num_inference_steps`` values linearly spaced on ``(0, 1]`` in
        descending order, applies the SD3 static shift warp, then appends a
        trailing ``0.0`` (the clean-data endpoint) -- so the length is
        ``num_inference_steps + 1``. The sampling loop consumes consecutive
        pairs ``(t[i], t[i+1])`` as ``(t, t_next)`` for :meth:`euler_step`.

        :param num_inference_steps: Number of Euler integration steps.
        :return: A float32 array of shape ``(num_inference_steps + 1,)``,
            strictly descending from near ``1.0`` to ``0.0``.
        """
        # Linspace on (0, 1] descending: 1, ..., 1/num (exclude 0 here; it is
        # appended after the shift warp so the terminal stays exactly 0.0).
        t = np.linspace(1.0, 0.0, num_inference_steps + 1, dtype=np.float64)[:-1]
        t = self._apply_shift(t)
        t = np.concatenate([t, np.array([0.0], dtype=np.float64)])
        return t.astype(np.float32)


logger.debug("FlowMatchEulerScheduler loaded (rectified-flow Euler, SD3 shift warp).")
