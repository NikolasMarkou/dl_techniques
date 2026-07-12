"""DDNM (Denoising Diffusion Null-space Model) driven by a BLIND bias-free denoiser.

Reference: Wang, Yu & Zhang, *"Zero-Shot Image Restoration Using Denoising Diffusion
Null-Space Model"* (ICLR 2023, arXiv:2212.00490).

Why this module exists
----------------------
DDNM — and every other post-2021 inverse-problem solver (DDRM arXiv:2201.11793, DPS
arXiv:2209.14687, PiGDM, DCDP) — is specified against a **noise-conditional** diffusion
network ``eps_theta(x_t, t)``: a model that is *told* the current noise level. A single
**blind** denoiser has no ``t`` input, and the literature contains **no published bridge**
between the two. That is why this repo's solver was stuck on the 2020 Kadkhodaie-Simoncelli
schedule.

**The bridge is exact degree-1 homogeneity.** Measured on
``convunext_denoiser_base_20260710_220452``: ``||D(a*y) - a*D(y)|| / ||a*D(y)||`` = **2.5e-05**,
and — decisively — **flat** across ``a`` in ``[0.12, 9.9]`` (an 80x range). A flat error is
float32 rounding noise; a genuine violation *grows* with ``a`` (a LayerNorm sibling checkpoint
shows 81-98%, and a bias-broken control here fires at 8.3e-01). Root cause:
``BiasFreeBatchNorm`` (no beta) + ``use_bias=False`` + ``leaky_relu``.

Homogeneity buys two things, and it is worth being precise about which:

1. **The x0-predictor.** DDNM needs, at every step, an estimate of the clean signal from the
   current iterate: ``x0_hat = D(x_t)``. A blind denoiser supplies this directly, with no ``t``
   input — it self-calibrates from the input's own noise level.
2. **Validity across the WHOLE schedule (this is the part that actually needs homogeneity).**
   An explicit diffusion schedule sweeps ``sigma`` over orders of magnitude, far outside any
   training curriculum. Degree-1 homogeneity means ``D(a*y) = a*D(y)`` *exactly*, so the
   denoiser's response co-scales with the noise level and the residual remains a valid score
   at **every** ``sigma_t`` — by construction, not by interpolation. Without it, the x0-estimate
   silently degrades at the ends of the schedule and the whole method quietly fails.
   The identity ``D_sigma(y) := sigma * D(y / sigma) == D(y)`` is therefore not a trick we
   apply; it is the *guarantee* that letting a blind net run the schedule is legitimate.
   :func:`homogeneity_error` checks it, and :class:`DDNMSolver` verifies it at construction.

What DDNM changes vs the K&S loop
---------------------------------
The K&S solver (``solver.py``) *self-calibrates* its noise level from the step magnitude
(``sigma_t = ||d_t|| / sqrt(N)``) and reaches data consistency only asymptotically, by
descending a gradient. DDNM instead:

* uses an **EXPLICIT geometric noise schedule** (no self-calibration), and
* imposes **HARD data consistency at every step** via the range-null decomposition::

      x0_hat  <-  A_dagger(y)  +  (I - A_dagger A) x0

  The measured (range-space) component is *replaced* by the observation exactly; only the
  unmeasured (null-space) component is filled in by the prior. Data consistency is therefore
  exact from iteration 1, rather than approached.

Because every operator here has orthonormal columns (``M^T M = I``, INV-4),
``A_dagger == adjoint`` and ``A_dagger A == project``, so the correction is exactly::

      x0_hat = operator.adjoint(measurements) + (x0 - operator.project(x0))

Noisy measurements
------------------
:class:`DDNMSolver` implements the **DDNM+** variant for noisy observations: the range-space
replacement is scaled by ``Sigma_t`` so that a measurement whose noise exceeds the current
diffusion noise level is not trusted absolutely. With ``measurement_sigma = 0`` this reduces
exactly to plain DDNM (full hard replacement).

Guardrails (inherited, non-negotiable)
--------------------------------------
The denoiser's Jacobian is **non-conservative** (asymmetry 0.14, ~800x a box-blur baseline) and
**not passive** (``||J||_2`` = 1.22-1.36). There is no global energy/log-density, so **no
calibrated-uncertainty claim is licensed** from this prior, and RED/PnP/MRED convergence
guarantees do not transfer. DDNM does not change that; it is a sampler, not a probability model.
"""

from typing import Any, Dict, Optional, Tuple

import keras
import numpy as np
import tensorflow as tf

from dl_techniques.utils.logger import logger

from .operators import MeasurementOperator

# Domain half-width of the [-0.5, +0.5] training convention (INV-1 / D-002).
_DOMAIN_HALF_WIDTH = 0.5

# Homogeneity is the load-bearing precondition of this whole module. This is the largest
# relative error we accept before refusing to run. Float32 rounding through a deep net lands
# at ~2.5e-05; a LayerNorm-normalized (non-homogeneous) checkpoint lands at ~0.9. The gap is
# ~4 orders of magnitude, so any threshold in between is safe. 1e-2 is deliberately generous.
_HOMOGENEITY_TOL = 1e-2


def homogeneity_error(
    prior: Any,
    y: "keras.KerasTensor",
    alphas: Tuple[float, ...] = (0.3, 1.3, 3.7, 6.1),
) -> Dict[float, float]:
    """Measure the degree-1 homogeneity error ``||D(a*y) - a*D(y)|| / ||a*D(y)||``.

    This is the precondition for using a BLIND denoiser to drive an explicit diffusion
    schedule (see the module docstring). Two design points matter:

    * The default ``alphas`` are deliberately **NOT powers of two**. Float32 scaling by a power
      of two is bit-exact, so powers of two report a spurious ``0.0`` regardless of whether the
      network is actually homogeneous. This masked the question on the first probe of this
      checkpoint and must not be repeated.
    * A *flat* error across a wide ``alpha`` range is rounding noise. A *growing* error is a real
      violation. Read the spread, not just the magnitude.

    Args:
        prior: An object exposing ``residual(y)`` and ``model`` (a ``DenoiserPrior``).
        y: A probe batch ``[B, H, W, C]`` in the ``[-0.5, +0.5]`` domain.
        alphas: Scale factors to test. Avoid powers of two.

    Returns:
        Mapping ``alpha -> relative error``.
    """
    denoise = getattr(prior, "model", prior)
    dy = np.asarray(denoise(y, training=False))
    out: Dict[float, float] = {}
    for a in alphas:
        da = np.asarray(denoise(keras.ops.multiply(float(a), y), training=False))
        num = float(np.linalg.norm(da - a * dy))
        den = float(np.linalg.norm(a * dy)) + 1e-12
        out[float(a)] = num / den
    return out


class DDNMSolver:
    """DDNM / DDNM+ sampler driven by a blind bias-free denoiser (arXiv:2212.00490).

    Drop-in alternative to :class:`~applications.bias_free_denoiser.solver.UniversalInverseSolver`
    with the SAME call signature (``solve(operator, measurements=..., shape=..., seed=...)``), so
    the two can be A/B'd on identical inputs. It consumes the same
    :class:`~applications.bias_free_denoiser.operators.MeasurementOperator` abstraction and adds
    no per-problem branching.

    The two structural differences from the K&S loop are an **explicit** geometric noise schedule
    (rather than one self-calibrated from the step magnitude) and **hard** range-space data
    consistency at every step (rather than asymptotic gradient-based consistency).

    Attributes:
        prior: A ``DenoiserPrior`` (must expose ``model`` / be callable, and ``residual``).
        steps: Number of diffusion steps ``T``.
        sigma_start: Initial (largest) noise level of the schedule.
        sigma_end: Final (smallest) noise level of the schedule.
        eta: DDIM stochasticity in ``[0, 1]``. ``0`` = deterministic DDIM; ``1`` = full ancestral.
        measurement_sigma: Noise std of the observation. ``0`` => plain DDNM (hard replacement);
            ``> 0`` => DDNM+ (``Sigma_t``-scaled replacement).
        verify_homogeneity: If ``True`` (default), probe degree-1 homogeneity on the first solve
            and REFUSE to run if it exceeds ``_HOMOGENEITY_TOL``. This is the module's central
            precondition; do not disable it casually.
    """

    def __init__(
        self,
        prior: Any,
        *,
        steps: int = 100,
        sigma_start: float = 0.5,
        sigma_end: float = 0.01,
        eta: float = 0.85,
        measurement_sigma: float = 0.0,
        verify_homogeneity: bool = True,
    ) -> None:
        """Configure the DDNM sampler.

        Args:
            prior: An object exposing ``residual(y)`` (e.g. a ``DenoiserPrior``).
            steps: Number of diffusion steps.
            sigma_start: Largest noise level (start of the anneal). The checkpoint's curriculum
                reaches ``sigma_max_end = 0.5``; homogeneity licenses going beyond it.
            sigma_end: Smallest noise level (end of the anneal).
            eta: DDIM stochasticity in ``[0, 1]``.
            measurement_sigma: Observation-noise std. ``0`` => plain DDNM.
            verify_homogeneity: Probe homogeneity on first solve and refuse if it fails.

        Raises:
            AttributeError: If ``prior`` exposes no callable ``residual``.
            ValueError: If ``steps < 1``, ``eta`` outside ``[0, 1]``, or the schedule is degenerate.
        """
        if not callable(getattr(prior, "residual", None)):
            raise AttributeError(
                "prior must expose a callable residual(y) method "
                "(e.g. DenoiserPrior); got %r" % type(prior).__name__
            )
        if int(steps) < 1:
            raise ValueError(f"steps must be >= 1; got {steps}")
        if not (0.0 <= float(eta) <= 1.0):
            raise ValueError(f"eta must be in [0, 1]; got {eta}")
        if not (0.0 < float(sigma_end) < float(sigma_start)):
            raise ValueError(
                f"require 0 < sigma_end < sigma_start; got {sigma_end}, {sigma_start}"
            )
        self.prior = prior
        self.steps = int(steps)
        self.sigma_start = float(sigma_start)
        self.sigma_end = float(sigma_end)
        self.eta = float(eta)
        self.measurement_sigma = float(measurement_sigma)
        self.verify_homogeneity = bool(verify_homogeneity)
        self._homogeneity_checked = False

        # Explicit GEOMETRIC noise schedule. This is the structural replacement for the K&S
        # loop's self-calibrated sigma_t (which Phase-2 measurement showed is the exact total
        # error scale, and whose retuning is worth +0.00 dB -- the gain has to come from
        # changing the solver FAMILY, not the schedule inside the old one).
        self.sigmas = np.geomspace(
            self.sigma_start, self.sigma_end, self.steps + 1
        ).astype(np.float32)

        logger.info(
            "DDNMSolver: steps=%d sigma %.3f -> %.3f (geometric) eta=%.2f "
            "measurement_sigma=%.4f mode=%s",
            self.steps, self.sigma_start, self.sigma_end, self.eta,
            self.measurement_sigma,
            "DDNM+ (noisy)" if self.measurement_sigma > 0 else "DDNM (noiseless)",
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _x0_estimate(self, x_t: "keras.KerasTensor") -> "keras.KerasTensor":
        """Estimate the clean signal ``x0`` from the current iterate ``x_t``.

        For a blind denoiser this is simply ``D(x_t)`` — no ``t`` input is needed, because the
        network self-calibrates from the input's own noise level. Exact degree-1 homogeneity is
        what makes this legitimate at EVERY ``sigma_t`` of the schedule, including levels far
        outside the training curriculum (module docstring, point 2).

        Equivalently: ``residual(x) = D(x) - x``, so ``D(x) = x + residual(x)``. We go through
        ``residual`` so any prior exposing only that interface still works.
        """
        return keras.ops.add(x_t, self.prior.residual(x_t))

    def _assert_homogeneous(self, y: "keras.KerasTensor") -> Dict[float, float]:
        """Probe homogeneity once and refuse to run if the precondition fails."""
        errs = homogeneity_error(self.prior, y)
        worst = max(errs.values())
        if worst > _HOMOGENEITY_TOL:
            raise RuntimeError(
                "DDNM requires a degree-1 homogeneous denoiser: it drives an EXPLICIT noise "
                "schedule spanning orders of magnitude, and only homogeneity guarantees the "
                "blind x0-estimate stays valid across it. Measured relative error %.3e "
                "(tolerance %.0e); per-alpha: %s.\n"
                "A bias-free ConvUNext is homogeneous ONLY with block_normalization='batchnorm' "
                "(BiasFreeBatchNorm), use_bias=False, and a positively-homogeneous activation "
                "(leaky_relu/relu). The factory-default 'gelu' and any LayerNorm block BREAK it. "
                "Pass verify_homogeneity=False to override at your own risk."
                % (worst, _HOMOGENEITY_TOL, {k: round(v, 8) for k, v in errs.items()})
            )
        logger.info(
            "DDNM homogeneity precondition PASS: worst rel. err %.3e over alphas %s "
            "(flat => float32 rounding, not violation)",
            worst, sorted(errs),
        )
        return errs

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
        """Run DDNM / DDNM+ for the given operator.

        Same signature as :meth:`UniversalInverseSolver.solve`, so the two are directly A/B-able.

        Args:
            operator: The measurement operator selecting the problem.
            measurements: The observation in the operator's measurement domain. ``None`` only for
                unconstrained prior sampling with a ``NullOperator`` (``shape`` then required).
            shape: Signal shape ``[B, H, W, C]``; REQUIRED when ``measurements`` is ``None``.
            seed: RNG seed for reproducibility.

        Returns:
            ``(x0, info)`` where ``info`` holds ``sigmas``, ``iterations``, ``constraint_errors``
            (``||measure(x) - measurements||`` per step, when measurements were given) and
            ``homogeneity``.

        Raises:
            ValueError: If ``measurements`` is ``None`` and ``shape`` is also ``None``.
            RuntimeError: If the homogeneity precondition fails (see class doc).
        """
        if seed is not None:
            tf.random.set_seed(seed)
            np.random.seed(seed)

        has_meas = measurements is not None
        if not has_meas:
            if shape is None:
                raise ValueError(
                    "measurements=None requires shape=(B,H,W,C) "
                    "(prior sampling with a NullOperator)"
                )
            measurements = keras.ops.zeros(shape)

        # --- Init: x_T = A_dagger(y) + sigma_T * eps. Starting from the adjoint (rather than
        # pure noise) gives DDNM a data-consistent starting point for free.
        mean = keras.ops.cast(operator.init_mean(measurements), "float32")
        x_t = keras.ops.add(
            mean,
            tf.random.normal(tf.shape(mean), stddev=float(self.sigmas[0]), dtype=tf.float32),
        )

        info: Dict[str, Any] = {
            "sigmas": [float(s) for s in self.sigmas],
            "iterations": [],
        }
        if has_meas:
            info["constraint_errors"] = []

        # Precondition check (once, on the real iterate — not a synthetic probe).
        if self.verify_homogeneity and not self._homogeneity_checked:
            info["homogeneity"] = self._assert_homogeneous(x_t)
            self._homogeneity_checked = True

        # A_dagger(y): the range-space component of the observation. Constant across steps, so
        # hoist it out of the loop.
        adj_y = keras.ops.cast(operator.adjoint(measurements), "float32") if has_meas else None

        for t in range(self.steps):
            sigma_t = float(self.sigmas[t])
            sigma_next = float(self.sigmas[t + 1])

            # (1) x0 estimate from the BLIND denoiser. Valid at this sigma_t by homogeneity.
            x0 = keras.ops.cast(self._x0_estimate(x_t), "float32")

            # (2) DDNM range-null correction (arXiv:2212.00490, Eq. 9):
            #         x0_hat = A_dagger(y) + (I - A_dagger A) x0
            #     With orthonormal columns (M^T M = I, INV-4): A_dagger == adjoint,
            #     A_dagger A == project. The measured component is REPLACED by the
            #     observation exactly; only the null space is filled by the prior.
            if has_meas:
                null_part = keras.ops.subtract(
                    x0, keras.ops.cast(operator.project(x0), "float32")
                )
                if self.measurement_sigma > 0.0:
                    # DDNM+ (Eq. 19): scale the range-space replacement so a measurement noisier
                    # than the current diffusion level is not trusted absolutely. lam -> 1 as the
                    # diffusion noise exceeds the measurement noise; -> 0 when it is far below.
                    lam = float(
                        min(1.0, (sigma_t ** 2) / (self.measurement_sigma ** 2 + 1e-12))
                    )
                    range_part = keras.ops.add(
                        keras.ops.multiply(lam, adj_y),
                        keras.ops.multiply(
                            1.0 - lam, keras.ops.cast(operator.project(x0), "float32")
                        ),
                    )
                else:
                    range_part = adj_y  # plain DDNM: exact hard replacement
                x0_hat = keras.ops.add(range_part, null_part)
            else:
                x0_hat = x0  # NullOperator => unconstrained prior sampling

            # (3) DDIM step down the EXPLICIT schedule to sigma_next.
            #     x_{t-1} = x0_hat + sqrt(sigma_next^2 - s^2) * dir + s * eps,
            #     where s = eta * sigma_next controls stochasticity. eta=0 is deterministic DDIM.
            s = self.eta * sigma_next
            det_scale = float(np.sqrt(max(sigma_next ** 2 - s ** 2, 0.0)))
            if sigma_t > 1e-12:
                # Direction pointing back to x_t (the DDIM "predicted noise" term).
                eps_hat = keras.ops.divide(
                    keras.ops.subtract(x_t, x0_hat), sigma_t
                )
            else:
                eps_hat = keras.ops.zeros_like(x_t)

            x_t = keras.ops.add(x0_hat, keras.ops.multiply(det_scale, eps_hat))
            if s > 0.0:
                x_t = keras.ops.add(
                    x_t,
                    tf.random.normal(tf.shape(x_t), stddev=s, dtype=tf.float32),
                )

            info["iterations"].append(t + 1)
            if has_meas:
                err = float(
                    keras.ops.sqrt(
                        keras.ops.sum(
                            keras.ops.square(
                                keras.ops.abs(
                                    keras.ops.subtract(operator.measure(x_t), measurements)
                                )
                            )
                        )
                    )
                )
                info["constraint_errors"].append(err)

            if (t + 1) % 25 == 0:
                logger.info("DDNM step %d/%d  sigma=%.4f", t + 1, self.steps, sigma_next)

        # Final x0 estimate + one last hard data-consistency projection, so the returned signal
        # satisfies the measurement exactly (the whole point of the null-space formulation).
        x0 = keras.ops.cast(self._x0_estimate(x_t), "float32")
        if has_meas:
            x0 = keras.ops.add(
                adj_y,
                keras.ops.subtract(x0, keras.ops.cast(operator.project(x0), "float32")),
            )
        info["stopped_iteration"] = self.steps
        logger.info("DDNM done after %d steps", self.steps)
        return x0, info
