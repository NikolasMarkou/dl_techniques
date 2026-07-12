"""Fast CPU tests for the reconstruction-quality harness (``metrics.py``, Step 1).

These tests validate that the PSNR/SSIM harness (a) guards identical images against
``+inf`` PSNR, (b) keeps SSIM bounded, (c) DISCRIMINATES quality (a cleaner image
out-PSNRs a noisier one against the same GT), and (d) drives the real forward-degrade
-> solve -> denorm pipeline to FINITE numbers for every task at a tiny budget.

They run on a tiny DETERMINISTIC stub denoiser (no 22 MB checkpoint), mirroring the
``_StubDenoiser`` pattern in ``test_solver.py`` but adding a ``denoise`` method the
harness needs for the single-pass denoise control. Images are ``>= 32 x 32`` so
``tf.image.ssim``'s default ``11 x 11`` window is valid.
"""

import keras
import numpy as np
import pytest

from applications.bias_free_denoiser import metrics

# Tiny CPU-fast problem: 32x32x3 (>= the 11x11 SSIM window; 32 divisible by 4 for SR).
IMAGE_SHAPE = (32, 32, 3)
FULL_SHAPE = (1, *IMAGE_SHAPE)

# All harness tasks (the six operators + the denoise control).
ALL_TASKS = (
    "denoise",
    "prior",
    "inpaint",
    "random_pixels",
    "super_resolution",
    "deblur",
    "compressive_sensing",
)


class _StubDenoiser:
    """Deterministic stand-in for :class:`DenoiserPrior` (no checkpoint).

    Exposes both ``residual(y) = alpha * (target - y)`` (the score estimate the
    solver consumes) and ``denoise(y) = y + residual(y) = D(y)`` (needed by the
    harness's single-pass denoise control). A linear contraction toward a fixed
    smooth image makes the ascent loop provably converge, isolating the harness.
    """

    def __init__(self, target: np.ndarray, alpha: float = 1.0) -> None:
        self.target = keras.ops.convert_to_tensor(target.astype(np.float32))
        self.alpha = float(alpha)

    def residual(self, y):
        return keras.ops.multiply(self.alpha, keras.ops.subtract(self.target, y))

    def denoise(self, y):
        y_t = keras.ops.convert_to_tensor(np.asarray(y, dtype=np.float32))
        return keras.ops.add(y_t, self.residual(y_t))


def _smooth_target() -> np.ndarray:
    """A smooth in-domain ``[1, 32, 32, 3]`` target in ``[0, 1]``."""
    h, w, c = IMAGE_SHAPE
    yy, xx = np.meshgrid(
        np.linspace(-0.4, 0.4, h), np.linspace(-0.4, 0.4, w), indexing="ij"
    )
    # Centered on the [0,1] domain center 0.5, half-amplitude 0.4 -> [0.1, 0.9].
    ramp = 0.5 + 0.5 * (yy + xx) / 0.8
    img = np.broadcast_to(ramp[None, :, :, None], FULL_SHAPE).astype(np.float32)
    return np.ascontiguousarray(img)


# ----------------------------------------------------------------------------------
# Metric primitives: capped PSNR, bounded SSIM, quality discrimination (SC2).
# ----------------------------------------------------------------------------------


class TestMetricPrimitives:
    def test_psnr_identical_is_capped_not_inf(self):
        """psnr(x, x) returns a large FINITE ceiling, never +inf."""
        x = _smooth_target()[0]  # already the [0, 1] domain (model == display)
        value = metrics.psnr(x, x)
        assert np.isfinite(value)
        assert value >= 99.0

    def test_ssim_in_unit_range(self):
        """0 <= ssim(x, y) <= 1 for two distinct [0, 1] images."""
        rng = np.random.default_rng(0)
        x = rng.uniform(0.0, 1.0, IMAGE_SHAPE).astype(np.float32)
        y = rng.uniform(0.0, 1.0, IMAGE_SHAPE).astype(np.float32)
        value = metrics.ssim(x, y)
        assert 0.0 <= value <= 1.0

    def test_psnr_discriminates_quality(self):
        """A cleaner image strictly out-PSNRs a noisier one against the same GT."""
        rng = np.random.default_rng(1)
        gt = _smooth_target()[0]  # already [0, 1]
        cleaner = np.clip(gt + rng.normal(0.0, 0.02, gt.shape), 0.0, 1.0)
        noisier = np.clip(gt + rng.normal(0.0, 0.20, gt.shape), 0.0, 1.0)
        assert metrics.psnr(cleaner, gt) > metrics.psnr(noisier, gt)

    def test_ssim_discriminates_quality(self):
        """A cleaner image scores higher SSIM than a noisier one (same GT)."""
        rng = np.random.default_rng(2)
        gt = _smooth_target()[0]  # already [0, 1]
        cleaner = np.clip(gt + rng.normal(0.0, 0.02, gt.shape), 0.0, 1.0)
        noisier = np.clip(gt + rng.normal(0.0, 0.20, gt.shape), 0.0, 1.0)
        assert metrics.ssim(cleaner, gt) > metrics.ssim(noisier, gt)


# ----------------------------------------------------------------------------------
# Single-task pipeline: forward-degrade -> solve/denoise -> denorm -> score (SC1).
# ----------------------------------------------------------------------------------


class TestDegradeAndReconstruct:
    @pytest.mark.parametrize("task", ALL_TASKS)
    def test_returns_finite_scores(self, task):
        """Every task yields finite PSNR/SSIM and a [0, 1] recon at a tiny budget."""
        target = _smooth_target()
        prior = _StubDenoiser(target)
        result = metrics.degrade_and_reconstruct(
            prior,
            target,
            task,
            solver_kwargs=dict(max_iterations=25, sigma_l=0.01, patience=15),
            seed=0,
        )
        assert result["task"] == task
        assert np.isfinite(result["psnr"]), f"{task}: non-finite PSNR"
        assert np.isfinite(result["ssim"]), f"{task}: non-finite SSIM"
        recon = result["recon01"]
        assert recon.shape == IMAGE_SHAPE
        assert np.all(np.isfinite(recon))
        assert recon.min() >= 0.0 and recon.max() <= 1.0


# ----------------------------------------------------------------------------------
# Full harness: tasks x configs -> finite table + per-config means (SC1/SC2).
# ----------------------------------------------------------------------------------


class TestRunHarness:
    def test_all_tasks_all_configs_finite(self):
        target = _smooth_target()
        prior = _StubDenoiser(target)
        configs = {
            "baseline": dict(max_iterations=25, h_max=0.1, patience=15),
            "fixed": dict(max_iterations=25, h_max=None, patience=15),
        }
        out = metrics.run_harness(prior, [target], list(ALL_TASKS), configs, seed=0)

        rows = out["rows"]
        assert len(rows) == len(ALL_TASKS) * len(configs)
        for r in rows:
            assert np.isfinite(r["psnr"]), f"{r}: non-finite PSNR"
            assert np.isfinite(r["ssim"]), f"{r}: non-finite SSIM"
            assert 0.0 <= r["ssim"] <= 1.0 + 1e-6

        means = out["config_means"]
        assert set(means) == set(configs)
        for cfg, m in means.items():
            assert np.isfinite(m["psnr"]), f"{cfg}: non-finite mean PSNR"
            assert np.isfinite(m["ssim"]), f"{cfg}: non-finite mean SSIM"
