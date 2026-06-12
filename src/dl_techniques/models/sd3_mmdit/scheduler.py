"""Rectified-flow Euler scheduler for SD3-style MMDiT training and sampling.

Faithful port of the Stable-Diffusion-3 ``FlowMatchEulerDiscreteScheduler``
semantics (rectified-flow / straight-line interpolation) plus the SD3 logit-
normal time sampling and the Eq.(19) loss weighting from the PyTorch reference.

Design note (framework-light, NOT a Keras layer)
------------------------------------------------
This is a plain Python object, NOT a ``keras.layers.Layer`` (nothing here is
trainable or serialized into a ``.keras`` graph). Two classes of methods:

* **In-graph tensor math** (``add_noise``, ``velocity_target``, ``euler_step``).
  Written with :mod:`keras.ops` so the trainer can call them inside a
  ``tf.function`` / on backend tensors. They are also valid on NumPy arrays
  (``keras.ops`` accepts array-likes), so the unit tests drive them with NumPy.
* **Host-side scheduling** (``sample_logit_normal_t``, ``logit_normal_weight``,
  ``timesteps``). These need the inverse standard-normal CDF (``ndtri``) and
  the logistic sigmoid (``expit``), which ``keras.ops`` has no backend-agnostic
  form of, so they run in NumPy via :mod:`scipy.special` (matching the PyTorch
  ``torch.special`` ops), in float64 internally and cast to float32 on return.

Convention (rectified flow / SD3)
---------------------------------
``t`` lives in ``[0, 1]``. ``t = 0`` is clean data ``x0``; ``t = 1`` is pure
noise ``x1``. The forward interpolation is the straight line::

    x_t = (1 - t) * x0 + t * noise

so the (constant-in-t) velocity target of the path ``dx/dt`` is::

    v = d/dt [ (1 - t) * x0 + t * noise ] = noise - x0

Reverse sampling integrates ``dx/dt = v`` from ``t = 1`` (noise) DOWN to
``t = 0`` (data); the Euler step uses ``dt = t_next - t`` which is NEGATIVE
during sampling. See :class:`FlowMatchEulerScheduler.euler_step` and D-007.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import keras
import numpy as np
from scipy.special import ndtri, expit

from dl_techniques.utils.logger import logger


@dataclass(frozen=True)
class FlowMatchEulerScheduler:
    """Rectified-flow Euler scheduler (forward noising + reverse Euler).

    A frozen dataclass holding the rectified-flow hyperparameters. The
    interpolation is a straight line between clean data (``t=0``) and pure
    noise (``t=1``); the velocity target is the constant ``noise - x0``.

    Args:
        num_train_timesteps: Number of discrete training timesteps (the
            continuous ``t`` is conceptually ``step / num_train_timesteps``).
            Used only by :meth:`timesteps` framing. Defaults to ``1000``.
        shift: SD3 static time-shift applied to the logit-normal samples and
            the inference time grid: ``t -> shift * t / (1 + (shift - 1) * t)``.
            ``shift = 1`` disables the warp. Defaults to ``3.0``.
        logit_mean: Mean of the logit-normal time-sampling distribution
            (pre-shift). Defaults to ``0.0``.
        logit_std: Standard deviation of the logit-normal time-sampling
            distribution. Defaults to ``1.0``.
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

        Args:
            x0: Clean data tensor (any shape).
            noise: Noise tensor broadcastable to ``x0`` (typically same shape).
            t: Time in ``[0, 1]``; a scalar or a tensor broadcastable to ``x0``
                (e.g. ``(B, 1, 1, 1)`` for a per-sample time against ``(B,...)``).

        Returns:
            The noised sample ``x_t`` with the broadcast shape of the inputs.
            At ``t = 0`` this is ``x0``; at ``t = 1`` this is ``noise``.
        """
        return (1.0 - t) * x0 + t * noise

    def velocity_target(self, x0, noise):
        """Rectified-flow velocity target ``noise - x0`` (constant in ``t``).

        This is the single source of truth for the straight-line velocity that
        the model is trained to predict. The trainer's MSE target and the
        sampler's Euler integrand both use this quantity.

        Args:
            x0: Clean data tensor.
            noise: Noise tensor broadcastable to ``x0``.

        Returns:
            The velocity ``noise - x0`` (same broadcast shape as the inputs).
        """
        return noise - x0

    def euler_step(self, x_t, v_pred, t, t_next):
        """One reverse Euler integration step of ``dx/dt = v``.

        Integrates ``x_next = x_t + v_pred * (t_next - t)``. During sampling the
        loop runs from ``t = 1`` (noise) DOWN to ``t = 0`` (data), so
        ``t_next < t`` and ``dt = t_next - t`` is NEGATIVE. For a straight-line
        rectified-flow path with the TRUE velocity this single step is exact:
        starting at ``x1`` and stepping ``t=1 -> t=0`` recovers ``x0`` exactly.

        Args:
            x_t: Current sample at time ``t``.
            v_pred: Predicted (or true) velocity at ``t`` (same shape as ``x_t``).
            t: Current time scalar/tensor in ``[0, 1]``.
            t_next: Next time scalar/tensor in ``[0, 1]`` (``< t`` when sampling).

        Returns:
            The integrated sample ``x_next`` at time ``t_next``.
        """
        # DECISION plan_2026-06-12_dfce0712/D-007: dt = t_next - t (signed),
        # NOT |t_next - t| and NOT t - t_next. Sampling descends t=1 -> t=0 so dt
        # is negative; with v = noise - x0 this makes one full step recover x0
        # exactly (x1 + (noise - x0)*(0 - 1) = noise - (noise - x0) = x0). A
        # positive/abs dt would integrate the wrong direction and diverge. See
        # decisions.md D-007.
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

        Args:
            batch_size: Number of times to draw.
            seed: Optional RNG seed for reproducibility.

        Returns:
            A float32 ``np.ndarray`` of shape ``(batch_size,)`` with values in
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
        ``(eps, 1 - eps)``. This method only COMPUTES the weight; the trainer
        is responsible for multiplying it into the per-sample loss (the HARD
        constraint keeps weighting out of the loss object).

        Args:
            t: Time(s) in ``(0, 1)``; a python float or a NumPy array-like.

        Returns:
            A float32 ``np.ndarray`` of weights (same shape as ``t``), positive
            and finite for ``t`` away from the boundaries.
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
        """Descending sampling time grid from ~1 down to 0.

        Builds ``num_inference_steps`` values linearly spaced on ``(0, 1]`` in
        DESCENDING order (start near 1, the noise end), applies the SD3 static
        shift warp, then APPENDS a trailing ``0.0`` (the clean-data endpoint).
        The sampling loop consumes consecutive pairs ``(t[i], t[i+1])`` as
        ``(t, t_next)`` for :meth:`euler_step`.

        Length is ``num_inference_steps + 1`` (the extra entry is the appended
        terminal ``0.0``).

        Args:
            num_inference_steps: Number of Euler integration steps.

        Returns:
            A float32 ``np.ndarray`` of shape ``(num_inference_steps + 1,)``,
            strictly descending, starting near ``1.0`` and ending at ``0.0``.
        """
        # Linspace on (0, 1] descending: 1, ..., 1/num (exclude 0 here; it is
        # appended after the shift warp so the terminal stays exactly 0.0).
        t = np.linspace(1.0, 0.0, num_inference_steps + 1, dtype=np.float64)[:-1]
        t = self._apply_shift(t)
        t = np.concatenate([t, np.array([0.0], dtype=np.float64)])
        return t.astype(np.float32)


logger.debug("FlowMatchEulerScheduler loaded (rectified-flow Euler, SD3 shift warp).")
