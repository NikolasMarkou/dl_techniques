"""CPU pytest for per-pixel additive-SURE maps and (later) conformal intervals.

Step 2 scope (this file, now): the per-pixel MC-SURE linear-toy self-check — the
Pre-Mortem STOP-IF gate for the spatial-map variants added to
``multiplicative_miyasawa.py`` (``hutchinson_divergence_map`` /
``additive_sure_risk_map``). Mirrors the scalar closed-form self-check in
``test_multiplicative_miyasawa.py`` (``D(y)=a*y``) but keeps the ``[H, W, C]`` map
instead of collapsing it to a single number.

Conformal-interval tests (construction + coverage sanity) arrive in Step 4 once
``conformal_denoiser_intervals.py`` exists; do NOT import that module here yet.

All RNG seeds are fixed and tensors are small -> fast on CPU (or GPU1 if a device
is visible; never GPU0, which hosts a live training job).
"""

import numpy as np
import tensorflow as tf

from dl_techniques.utils.multiplicative_miyasawa import (
    hutchinson_divergence_map,
    additive_sure_risk_map,
)


class TestPerPixelSureMap:
    """Pre-Mortem STOP-IF: per-pixel SURE maps must match the D(y)=a*y closed form.

    If either assertion fails, the spatial-unreduce in Step 1 broke the estimator
    scale — do NOT loosen these tolerances; report actual-vs-expected so the
    per-pixel reduction can be re-derived (plan Pre-Mortem, mirrors the scalar gate
    ``test_multiplicative_miyasawa.py::TestAdditiveSureSelfCheck``).
    """

    def test_divergence_map_linear_toy(self):
        # D(y) = a*y has diagonal Jacobian dD_i/dy_i = a at EVERY element, so the
        # per-pixel divergence map is ~= a everywhere. With Rademacher probes v the
        # elementwise product v * jvp = v * (a*v) = a * v^2 = a exactly (v^2 == 1),
        # so the map is a up to fp regardless of eps / probe count.
        a = 0.7
        rng = np.random.default_rng(0)
        y = tf.constant(rng.uniform(-0.5, 0.5, size=(8, 16, 16, 3)).astype(np.float32))

        div_map = hutchinson_divergence_map(
            lambda t: a * t, y, n_hutchinson=64, eps=1e-3, rademacher=True, seed=0,
        )
        div_np = div_map.numpy()
        assert div_np.shape == (16, 16, 3)

        max_dev = float(np.max(np.abs(div_np - a)))
        mean = float(np.mean(div_np))
        print(f"\n[div map] mean={mean:.5f} exact={a} max|dev|={max_dev:.2e}")
        # Linear map + Rademacher probes -> exact per element up to fp; tight tol.
        assert max_dev <= 0.02, f"div map deviates from a={a} by {max_dev:.3e}"

    def test_sure_map_linear_toy_matches_analytic(self):
        # x ~ U[-0.5, 0.5] (so E[x^2] = 1/12), y = x + N(0, sigma^2). Denoiser D(y)=a*y.
        # Per pixel, D(y) - x = (a-1)*x + a*e (e ~ N(0, sigma^2)), so the true per-pixel
        # MSE is E[(D(y)-x)^2] = (a-1)^2 x^2 + a^2 sigma^2 (cross term zero in expectation).
        # The SURE map is unbiased for this per element; compare its MEAN over the map
        # (batch already reduced) to the analytic mean E_x[(a-1)^2 x^2] + a^2 sigma^2
        # = (a-1)^2 / 12 + a^2 sigma^2. Per-pixel MC is high variance, so we compare the
        # spatial/batch MEAN, NOT element-wise values.
        a = 0.6
        sigma = 0.3
        B, H, W, C = 32, 16, 16, 3
        rng = np.random.default_rng(0)
        x = rng.uniform(-0.5, 0.5, size=(B, H, W, C)).astype(np.float32)
        e = rng.normal(0.0, sigma, size=(B, H, W, C)).astype(np.float32)
        y = tf.constant(x + e)

        sure_map = additive_sure_risk_map(
            lambda t: a * t, y, sigma=sigma, n_hutchinson=64, eps=1e-3, seed=0,
        )
        sure_np = sure_map.numpy()
        assert sure_np.shape == (H, W, C)

        est_mean = float(np.mean(sure_np))
        analytic = (a - 1.0) ** 2 * (1.0 / 12.0) + a ** 2 * sigma ** 2
        # Realized per-draw MSE over the same draws (lower-variance reference).
        realized_mse = float(np.mean((a * (x + e) - x) ** 2))
        print(
            f"\n[sure map] est_mean={est_mean:.5f} analytic={analytic:.5f} "
            f"realized_mse={realized_mse:.5f} |est-analytic|={abs(est_mean - analytic):.5f}"
        )
        # div_map is exact for Rademacher+linear; the residual mean over B*H*W*C
        # samples is tight, so a generous 0.01 atol has wide headroom.
        assert abs(est_mean - analytic) <= 0.01, (
            f"SURE map mean {est_mean:.5f} vs analytic {analytic:.5f} "
            f"(realized {realized_mse:.5f}) exceeds MC tol 0.01"
        )


# ---------------------------------------------------------------------------
# Step 4 (deps: 3) will add conformal-interval tests HERE once
# ``dl_techniques.utils.conformal_denoiser_intervals`` exists:
#   * construction / import test (API callables present)
#   * coverage sanity: synthetic exchangeable residuals reach ~target coverage
#     after per-sigma (Mondrian) split-conformal calibration.
# Do NOT import the conformal module until Step 3 has created it.
# ---------------------------------------------------------------------------
