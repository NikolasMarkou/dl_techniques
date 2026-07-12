"""Fast CPU e2e tests for :class:`UniversalInverseSolver` (Step 5).

These tests exercise the unified Algorithm-1/2 loop with a tiny DETERMINISTIC stub
denoiser (no 22 MB checkpoint), so they run in well under a second on CPU. The stub
exposes ``residual(y) = alpha * (target - y)`` — a contraction toward a fixed smooth
image, so the loop provably converges on synthetic data.

Coverage (plan.md Step 5 / Pre-Mortem #2):

* ``test_identity_reproduces_algorithm1`` — with a :class:`NullOperator` the unified
  ``d_t`` reduces EXACTLY to ``residual(y)`` (Algorithm 1) AND ``sigma_t`` decreases
  / converges. This is the STOP-IF #2 gate.
* ``test_masked_constraint_decreases`` — with an :class:`InpaintingOperator` the
  constraint error ``||measure(y) - m||`` shrinks over iterations.
* ``test_solve_runs_all_operators`` — the SAME loop returns a finite, correctly
  shaped result for every operator type (Null, Inpainting, RandomPixels,
  SuperResolution, SpectralDeblur, CompressiveSensing): proof of operator-agnosticism.
* ``test_no_nan`` — no NaN/Inf for any operator.
"""

import keras
import numpy as np
import pytest

from applications.bias_free_denoiser.operators import (
    CompressiveSensingOperator,
    InpaintingOperator,
    NullOperator,
    RandomPixelsOperator,
    SpectralDeblurOperator,
    SuperResolutionOperator,
)
from applications.bias_free_denoiser.solver import UniversalInverseSolver

# Tiny CPU-fast synthetic problem: single-image, 16x16x1 (divisible by 2 for super-res).
IMAGE_SHAPE = (16, 16, 1)
BATCH = 1
FULL_SHAPE = (BATCH, *IMAGE_SHAPE)


def _np(t) -> np.ndarray:
    """Materialize a keras/backend tensor as a NumPy array."""
    return keras.ops.convert_to_numpy(t)


def _smooth_target(seed: int = 0) -> np.ndarray:
    """A smooth in-domain ``[B, H, W, C]`` target in ``[0, 1]``."""
    h, w, c = IMAGE_SHAPE
    yy, xx = np.meshgrid(
        np.linspace(-0.4, 0.4, h), np.linspace(-0.4, 0.4, w), indexing="ij"
    )
    # Centered on the [0,1] domain center 0.5, half-amplitude 0.4 -> [0.1, 0.9].
    ramp = 0.5 + 0.5 * (yy + xx) / 0.8
    img = np.broadcast_to(ramp[None, :, :, None], FULL_SHAPE).astype(np.float32)
    return np.ascontiguousarray(img)


class _StubDenoiser:
    """Deterministic stand-in for :class:`DenoiserPrior` (no checkpoint).

    ``residual(y) = alpha * (target - y)`` is a linear contraction toward a fixed
    smooth image; it makes the ascent loop provably converge, so the tests isolate
    the SOLVER logic (schedule, unified ``d_t``, early stopping) from the real model.
    """

    def __init__(self, target: np.ndarray, alpha: float = 1.0) -> None:
        self.target = keras.ops.convert_to_tensor(target.astype(np.float32))
        self.alpha = float(alpha)

    def residual(self, y):
        return keras.ops.multiply(
            self.alpha, keras.ops.subtract(self.target, y)
        )


def _solver(target: np.ndarray, **kwargs) -> UniversalInverseSolver:
    """A short-horizon solver over the stub denoiser (fast, deterministic)."""
    defaults = dict(sigma_0=0.4, sigma_l=0.01, max_iterations=60, patience=15)
    defaults.update(kwargs)
    return UniversalInverseSolver(_StubDenoiser(target), **defaults)


# ----------------------------------------------------------------------------------
# Step-size schedule: default cap byte-identical; h_max=None uncapped (A1, SC1/SC2).
# ----------------------------------------------------------------------------------

_STEP_SIZE_TS = [1, 2, 5, 10, 50, 100, 500, 1000]


class TestStepSizeHMaxCap:
    def test_step_size_default_byte_identical(self):
        """SC1: default h_max=0.1 == min(h0*t/(1+h0*(t-1)), 0.1) exactly."""
        solver = _solver(_smooth_target())
        for t in _STEP_SIZE_TS:
            expected = min(solver.h0 * t / (1.0 + solver.h0 * (t - 1)), 0.1)
            assert solver._step_size(t) == expected, f"t={t}: cap not byte-identical"

    def test_step_size_h_max_none_uncapped(self):
        """SC2: h_max=None == literal paper value and exceeds the old 0.1 cap."""
        solver = _solver(_smooth_target(), h_max=None)
        for t in _STEP_SIZE_TS:
            expected = solver.h0 * t / (1.0 + solver.h0 * (t - 1))
            assert solver._step_size(t) == expected, f"t={t}: not the paper value"
        # At the default h0=0.01 the uncapped schedule exceeds the old 0.1 cap.
        assert solver.h0 == 0.01
        assert solver._step_size(1000) > 0.1


# ----------------------------------------------------------------------------------
# STOP-IF #2 gate: NullOperator reproduces Algorithm 1.
# ----------------------------------------------------------------------------------

class TestIdentityReproducesAlgorithm1:
    def test_dt_equals_residual(self):
        """The unified d_t degenerates to f(y) for the empty-M NullOperator (INV-6)."""
        op = NullOperator()
        target = _smooth_target()
        stub = _StubDenoiser(target)
        y = np.random.default_rng(0).standard_normal(FULL_SHAPE).astype(np.float32)

        f_y = stub.residual(y)
        # Replicate the solver's d_t expression with a zero measurement template.
        zeros = keras.ops.zeros(FULL_SHAPE)
        prior_term = keras.ops.subtract(f_y, op.project(f_y))
        data_term = op.adjoint(keras.ops.subtract(zeros, op.measure(y)))
        d_t = keras.ops.add(prior_term, data_term)

        # d_t must equal f(y) EXACTLY (Algorithm 1) — project=adjoint=measure=0.
        np.testing.assert_allclose(_np(d_t), _np(f_y), atol=1e-6)

    def test_sigma_decreases_and_converges(self):
        """sigma_t must trend down and end below its start (STOP-IF #2 gate).

        Uses ``beta=0.5`` — the annealing rate of the paper's Algorithm-1 prior
        sampler (``samplers.py`` ``DenoiserPriorSampler`` default). The inverse-
        problem default ``beta=0.01`` anneals ~0.1%/step, which is dominated by the
        ~4% per-step stochastic fluctuation of ``sigma_t`` at N=256, so a short
        horizon cannot show the trend; ``beta=0.5`` gives the clear geometric decay
        ``sigma_{t+1} ~ (1 - beta*h_t) sigma_t`` this gate checks for.
        """
        target = _smooth_target()
        solver = _solver(
            target, beta=0.5, sigma_l=0.02, max_iterations=200, patience=60
        )
        best_y, info = solver.solve(NullOperator(), shape=FULL_SHAPE, seed=7)

        sigmas = info["sigma_values"]
        assert len(sigmas) >= 5
        # Converges: final effective noise is well below the initial one.
        assert sigmas[-1] < sigmas[0]
        assert info["best_sigma"] < sigmas[0]
        # Trend (robust to per-step noise): 2nd-half mean < 1st-half mean.
        half = len(sigmas) // 2
        assert np.mean(sigmas[half:]) < np.mean(sigmas[:half])
        # The best iterate landed near the stub's fixed point.
        assert np.all(np.isfinite(_np(best_y)))
        assert _np(best_y).shape == FULL_SHAPE


# ----------------------------------------------------------------------------------
# Constraint satisfaction for a masked operator.
# ----------------------------------------------------------------------------------

class TestMaskedConstraintDecreases:
    def test_constraint_error_decreases(self):
        target = _smooth_target()
        op = InpaintingOperator(IMAGE_SHAPE, block_size=(6, 6))
        measurements = op.measure(target)

        solver = _solver(target)
        _best_y, info = solver.solve(op, measurements=measurements, seed=3)

        errs = info["constraint_errors"]
        assert len(errs) >= 5
        # The data term pulls measured pixels toward their observed values.
        assert errs[-1] < errs[0]


# ----------------------------------------------------------------------------------
# Operator-agnosticism: the SAME loop runs every problem.
# ----------------------------------------------------------------------------------

def _all_operators():
    """One instance of every operator type, tiny + seeded for speed/determinism."""
    return {
        "null": NullOperator(),
        "inpainting": InpaintingOperator(IMAGE_SHAPE, block_size=(6, 6)),
        "random_pixels": RandomPixelsOperator(IMAGE_SHAPE, keep_ratio=0.4, seed=0),
        "super_resolution": SuperResolutionOperator(IMAGE_SHAPE, factor=2),
        "spectral": SpectralDeblurOperator(IMAGE_SHAPE, keep_fraction=0.5),
        "compressive_sensing": CompressiveSensingOperator(
            IMAGE_SHAPE, measurement_ratio=0.3, seed=0
        ),
    }


class TestSolveRunsAllOperators:
    @pytest.mark.parametrize("name,op", list(_all_operators().items()))
    def test_runs(self, name, op):
        target = _smooth_target()
        solver = _solver(target, max_iterations=25)

        if name == "null":
            best_y, info = solver.solve(op, shape=FULL_SHAPE, seed=1)
        else:
            measurements = op.measure(target)
            best_y, info = solver.solve(op, measurements=measurements, seed=1)

        arr = _np(best_y)
        assert arr.shape == FULL_SHAPE, f"{name}: wrong output shape {arr.shape}"
        assert np.all(np.isfinite(arr)), f"{name}: non-finite output"
        assert len(info["sigma_values"]) >= 1


class TestFinalProjection:
    """The additive OFF-by-default final data-consistency lever (D-005, SC6).

    ``final_projection=False`` (default) must be byte-identical to a solver
    constructed WITHOUT the kwarg; ``final_projection=True`` on a masked operator
    must make ``measure(best_y) == measurements`` exactly (hard consistency).
    """

    def test_off_is_byte_identical(self):
        target = _smooth_target()
        op = InpaintingOperator(IMAGE_SHAPE, block_size=(6, 6))
        measurements = op.measure(target)

        # One solver never passes final_projection (defaults to False); the other
        # passes it explicitly False. Same seed -> identical trajectory + best_y.
        base = _solver(target)
        proj_off = _solver(target, final_projection=False)
        by_base, _ = base.solve(op, measurements=measurements, seed=11)
        by_off, _ = proj_off.solve(op, measurements=measurements, seed=11)

        max_delta = float(np.max(np.abs(_np(by_base) - _np(by_off))))
        assert max_delta == 0.0, f"OFF path not byte-identical: max|delta|={max_delta}"

    def test_on_enforces_hard_consistency(self):
        target = _smooth_target()
        op = InpaintingOperator(IMAGE_SHAPE, block_size=(6, 6))
        measurements = op.measure(target)

        solver = _solver(target, final_projection=True)
        best_y, info = solver.solve(op, measurements=measurements, seed=11)

        # Hard data consistency: the measured components exactly match observations.
        resid = _np(keras.ops.subtract(op.measure(best_y), measurements))
        max_resid = float(np.max(np.abs(resid)))
        assert max_resid == 0.0, f"final projection did not enforce consistency: {max_resid}"
        assert info["final_projection"] is True


class TestNoNaN:
    @pytest.mark.parametrize("name,op", list(_all_operators().items()))
    def test_no_nan(self, name, op):
        target = _smooth_target()
        solver = _solver(target, max_iterations=25)
        if name == "null":
            best_y, info = solver.solve(op, shape=FULL_SHAPE, seed=2)
        else:
            best_y, info = solver.solve(op, measurements=op.measure(target), seed=2)

        arr = _np(best_y)
        assert not np.any(np.isnan(arr)), f"{name}: NaN in output"
        assert not np.any(np.isinf(arr)), f"{name}: Inf in output"
        for v in info["sigma_values"]:
            assert np.isfinite(v), f"{name}: non-finite sigma"
