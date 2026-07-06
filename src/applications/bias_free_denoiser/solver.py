"""Universal inverse-problem solver for the bias-free denoiser prior.

This module holds the single integration point of the app: :class:`UniversalInverseSolver`.
ONE coarse-to-fine stochastic gradient-ascent loop (Kadkhodaie & Simoncelli 2020,
Algorithm 2) solves EVERY problem — prior sampling, inpainting, random-missing
pixels, super-resolution, spectral deblurring, compressive sensing. The problem is
selected purely by which :class:`~applications.bias_free_denoiser.operators.MeasurementOperator`
is passed to :meth:`UniversalInverseSolver.solve`; the loop contains NO per-problem
branching (INV-6 — the whole point of the plan, D-001).

Unified update (INV-6)
----------------------
At each iteration the ascent direction is::

    d_t = (I - M M^T) f(y) + M (x_c - M^T y)
        = (f(y) - project(f(y))) + adjoint(measurements - measure(y))

where ``f(y) = D(y) - y`` is the denoiser residual (the implicit-prior score
estimate, from :meth:`DenoiserPrior.residual`). For the empty-measurement
:class:`NullOperator` every ``project`` / ``adjoint`` / ``measure`` returns zeros,
so ``d_t`` degenerates EXACTLY to ``d_t = f(y)`` — the unconstrained Algorithm-1
prior sampler. This degeneration is the Step-5 STOP-IF gate (Pre-Mortem #2) and is
verified end-to-end in ``test_solver.py``.

Schedule (A7 — reused verbatim from the paper-exact ``samplers.py:69-141``)
-------------------------------------------------------------------------
* step size ``h_t = min(h0 * t / (1 + h0 * (t - 1)), 0.1)``
* effective noise ``sigma_t = ||d_t|| / sqrt(N) = sqrt(mean(d_t^2))``
* injected noise ``gamma_t^2 = max(((1 - beta*h_t)^2 - (1 - h_t)^2) * sigma_t^2, 0)``
* update ``y <- y + h_t * d_t + gamma_t * z_t``, ``z_t ~ N(0, I)``
* early stopping with patience (best-``sigma`` tracking).

Domain (INV-1 / D-002 / S1)
---------------------------
Pixels live in ``[-0.5, +0.5]`` with center ``c0 = 0.0``. Unlike the old
``samplers.py`` (which hard-clipped every iterate to ``[-1, +1]``), interior
clipping is OPTIONAL and OFF by default: F1 §3 established that clipping breaks the
``residual = score`` identity at the domain boundary (S1). With a domain-appropriate
``sigma_0`` the iterates stay interior naturally, so no clip is needed; the ``clip``
flag remains available for callers that prefer a hard guard.
"""

from typing import Any, Dict, Optional, Tuple

import keras
import numpy as np
import tensorflow as tf

from dl_techniques.utils.logger import logger

from .operators import MeasurementOperator

# Domain-adjusted default initial noise (INV-1 / D-002). The paper used sigma_0 = 1.0
# for [0, 1] data (a full-unit range centered at 0.5). THIS checkpoint trains in the
# half-unit [-0.5, +0.5] range (center 0.0) and its curriculum topped out at sigma =
# 0.25, so a coarse-but-in-domain start is ~0.4 (well below the 0.5 half-width, so the
# initial N(c0, sigma_0^2) field rarely lands outside the domain). Overridable.
_DEFAULT_SIGMA_0 = 0.4

# Domain half-width for the OPTIONAL interior clip (INV-1). Off by default (see below).
_DEFAULT_CLIP_RANGE = (-0.5, 0.5)


class UniversalInverseSolver:
    """One Algorithm-1/2 loop that solves all six inverse problems (INV-6).

    Wraps a denoiser prior and runs the paper's coarse-to-fine stochastic gradient
    ascent. The measurement operator is the ONLY thing that varies between problems
    — the solver never branches on a problem type.

    Attributes:
        prior: An object exposing ``residual(y) -> D(y) - y`` (typically a
            :class:`~applications.bias_free_denoiser.denoiser_prior.DenoiserPrior`).
        sigma_0: Initial noise std of the Algorithm-2 init (domain-adjusted default
            ``0.4`` for the ``[-0.5, +0.5]`` model — see module note / D-002).
        sigma_l: Stopping threshold on the effective noise ``sigma_t``.
        h0: Step-size schedule parameter.
        beta: Noise-injection parameter (``0.01`` for inverse problems, paper §3.2).
        max_iterations: Hard cap on iterations.
        patience: Iterations without a ``sigma_t`` improvement before early stop.
        clip: If ``True``, clip each iterate to ``clip_range`` (OFF by default; the
            hard clip breaks residual=score at the boundary, S1).
        clip_range: ``(lo, hi)`` interior clip bounds used only when ``clip`` is set.
    """

    def __init__(
        self,
        prior: Any,
        *,
        sigma_0: float = _DEFAULT_SIGMA_0,
        sigma_l: float = 0.01,
        h0: float = 0.01,
        beta: float = 0.01,
        max_iterations: int = 1000,
        patience: int = 20,
        clip: bool = False,
        clip_range: Tuple[float, float] = _DEFAULT_CLIP_RANGE,
    ) -> None:
        """Configure the solver.

        Args:
            prior: An object exposing ``residual(y)`` (e.g. a ``DenoiserPrior``).
            sigma_0: Initial noise std (default ``0.4``, domain-adjusted, D-002).
            sigma_l: Effective-noise stopping threshold (default ``0.01``).
            h0: Step-size schedule parameter (default ``0.01``).
            beta: Noise-injection parameter (default ``0.01``, paper §3.2 for
                inverse problems).
            max_iterations: Iteration cap (default ``1000``).
            patience: No-improvement iterations before early stop (default ``20``).
            clip: Enable the OPTIONAL interior clip (default ``False``; see class
                doc / S1 for why clipping is off by default).
            clip_range: ``(lo, hi)`` bounds for the optional clip (default
                ``(-0.5, +0.5)``).

        Raises:
            AttributeError: If ``prior`` does not expose a callable ``residual``.
        """
        if not callable(getattr(prior, "residual", None)):
            raise AttributeError(
                "prior must expose a callable residual(y) method "
                "(e.g. DenoiserPrior); got %r" % type(prior).__name__
            )
        self.prior = prior
        self.sigma_0 = float(sigma_0)
        self.sigma_l = float(sigma_l)
        self.h0 = float(h0)
        self.beta = float(beta)
        self.max_iterations = int(max_iterations)
        self.patience = int(patience)
        self.clip = bool(clip)
        self.clip_range = (float(clip_range[0]), float(clip_range[1]))
        logger.info(
            "UniversalInverseSolver: sigma_0=%.3f sigma_l=%.3f h0=%.3f beta=%.3f "
            "max_iter=%d patience=%d clip=%s (domain [-0.5,+0.5])",
            self.sigma_0, self.sigma_l, self.h0, self.beta,
            self.max_iterations, self.patience, self.clip,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _step_size(self, t: int) -> float:
        """Paper-exact adaptive step ``h_t = min(h0 t / (1 + h0 (t-1)), 0.1)`` (A7)."""
        return min(self.h0 * t / (1.0 + self.h0 * (t - 1)), 0.1)

    @staticmethod
    def _l2(t: "keras.KerasTensor") -> float:
        """Euclidean norm ``sqrt(sum(|t|^2))`` (``abs`` handles complex tensors)."""
        mag2 = keras.ops.square(keras.ops.abs(t))
        return float(keras.ops.sqrt(keras.ops.sum(mag2)))

    def _maybe_clip(self, y: "keras.KerasTensor") -> "keras.KerasTensor":
        """Apply the OPTIONAL interior clip.

        Off by default: the old ``samplers.py`` clipped every iterate to ``[-1, 1]``,
        but F1 §3 showed a hard clip breaks the ``residual = score`` identity at the
        domain boundary (S1). With a domain-appropriate ``sigma_0`` iterates stay
        interior on their own, so clipping is opt-in, not the default (INV-1).
        """
        if not self.clip:
            return y
        return keras.ops.clip(y, self.clip_range[0], self.clip_range[1])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(
        self,
        operator: MeasurementOperator,
        measurements: Optional[Any] = None,
        shape: Optional[Tuple[int, ...]] = None,
        seed: Optional[int] = None,
    ) -> Tuple["keras.KerasTensor", Dict[str, Any]]:
        """Run the unified Algorithm-1/2 loop for the given operator.

        The SAME loop serves every problem — only ``operator`` changes (INV-6).
        For a :class:`NullOperator` the update degenerates to Algorithm-1 prior
        sampling (``d_t = f(y)``) and ``measurements`` may be ``None`` (``shape`` is
        then required to size the initial noise field).

        Args:
            operator: The measurement operator selecting the problem. Its
                ``measure`` / ``adjoint`` / ``project`` / ``init_mean`` methods are
                the only problem-specific code the solver touches.
            measurements: The observation in the operator's measurement domain
                (same-shape masked signal for mask/super-res-scaled/spectral/CS
                operators — the operator's methods own the domain). ``None`` only
                for prior sampling with a ``NullOperator``.
            shape: Signal shape ``[B, H, W, C]``. REQUIRED when ``measurements`` is
                ``None`` (used to build the init noise field); ignored otherwise.
            seed: Optional RNG seed for reproducible init + injected noise.

        Returns:
            ``(best_y, info)`` where ``best_y`` is the signal-domain iterate with the
            smallest observed ``sigma_t`` and ``info`` holds ``iterations``,
            ``sigma_values``, ``best_sigma``, ``stopped_iteration`` and — when
            ``measurements`` was supplied — ``constraint_errors``
            (``||measure(y) - measurements||`` per iteration).

        Raises:
            ValueError: If ``measurements`` is ``None`` and ``shape`` is also ``None``.
        """
        if seed is not None:
            tf.random.set_seed(seed)
            np.random.seed(seed)

        has_measurements = measurements is not None
        if not has_measurements:
            if shape is None:
                raise ValueError(
                    "measurements=None requires shape=(B,H,W,C) "
                    "(prior sampling with a NullOperator)"
                )
            # Measurement-domain template of zeros; for NullOperator the measurement
            # domain equals the signal domain, so init_mean/measure/adjoint all treat
            # this as the signal-shaped zero field (Algorithm-1 init N(c0, sigma_0^2)).
            measurements = keras.ops.zeros(shape)

        # --- Algorithm-2 init (F2): y0 = init_mean(measurements) + N(0, sigma_0^2 I).
        mean = keras.ops.cast(operator.init_mean(measurements), "float32")
        y = keras.ops.add(
            mean,
            tf.random.normal(tf.shape(mean), stddev=self.sigma_0, dtype=tf.float32),
        )
        y = self._maybe_clip(y)

        # --- Early-stopping / convergence bookkeeping (A7, mirrors samplers.py).
        best_sigma = float("inf")
        best_y = y
        patience_counter = 0
        info: Dict[str, Any] = {"iterations": [], "sigma_values": []}
        if has_measurements:
            info["constraint_errors"] = []
        sigma_prev = self.sigma_0
        t = 1

        logger.info("solving with %s (patience=%d)...", type(operator).__name__, self.patience)
        while sigma_prev > self.sigma_l and t <= self.max_iterations:
            h_t = self._step_size(t)

            # --- Unified ascent direction d_t (INV-6). NO per-problem branching:
            #     d_t = (f(y) - project(f(y))) + adjoint(measurements - measure(y)).
            # NullOperator -> project=adjoint=measure=0 -> d_t = f(y) (Algorithm-1).
            f_y = self.prior.residual(y)
            prior_term = keras.ops.subtract(f_y, operator.project(f_y))
            residual_meas = keras.ops.subtract(measurements, operator.measure(y))
            data_term = keras.ops.cast(operator.adjoint(residual_meas), "float32")
            d_t = keras.ops.add(prior_term, data_term)

            # --- Effective noise sigma_t = ||d_t|| / sqrt(N) = sqrt(mean(d_t^2)) (A7).
            sigma_t_sq = keras.ops.mean(keras.ops.square(d_t))
            sigma_t = float(keras.ops.sqrt(keras.ops.maximum(sigma_t_sq, 1e-20)))

            # --- Early stopping: keep the best (lowest-sigma) iterate.
            if sigma_t < best_sigma:
                best_sigma = sigma_t
                best_y = y
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= self.patience:
                logger.info(
                    "early stop at iter %d (%d steps without improvement)",
                    t, self.patience,
                )
                break

            # --- Noise injection amplitude gamma_t (A7, paper Algorithm 2).
            beta_h = min(self.beta * h_t, 0.99)
            gamma_sq = ((1.0 - beta_h) ** 2 - (1.0 - h_t) ** 2) * sigma_t ** 2
            gamma_t = float(np.sqrt(max(gamma_sq, 0.0)))
            z_t = tf.random.normal(tf.shape(y), dtype=tf.float32)

            # --- Update y <- y + h_t d_t + gamma_t z_t.
            step = keras.ops.add(
                keras.ops.multiply(h_t, d_t),
                keras.ops.multiply(gamma_t, z_t),
            )
            y = self._maybe_clip(keras.ops.add(y, step))

            info["iterations"].append(t)
            info["sigma_values"].append(sigma_t)
            if has_measurements:
                err = self._l2(
                    keras.ops.subtract(operator.measure(y), measurements)
                )
                info["constraint_errors"].append(err)

            if t % 50 == 0:
                logger.info(
                    "iter %d: sigma=%.6f best=%.6f patience=%d/%d",
                    t, sigma_t, best_sigma, patience_counter, self.patience,
                )
            sigma_prev = sigma_t
            t += 1

        info["best_sigma"] = best_sigma
        info["stopped_iteration"] = t - 1
        logger.info(
            "done after %d iterations; best sigma=%.6f", t - 1, best_sigma,
        )
        return best_y, info
