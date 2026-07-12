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

import pytest
import numpy as np
import tensorflow as tf

from dl_techniques.utils.multiplicative_miyasawa import (
    hutchinson_divergence_map,
    additive_sure_risk_map,
)
from dl_techniques.utils.conformal_denoiser_intervals import (
    DOMAIN_MIN,
    DOMAIN_MAX,
    conformal_quantile,
    calibrate_per_sigma,
    predict_intervals,
    evaluate_coverage,
    _unwrap_point_estimate,
)


class TestPerPixelSureMap:
    """Pre-Mortem STOP-IF: per-pixel SURE maps must match the D(y)=a*y closed form.

    If either assertion fails, the spatial-unreduce in Step 1 broke the estimator
    scale — do NOT loosen these tolerances; report actual-vs-expected so the
    per-pixel reduction can be re-derived (plan Pre-Mortem, mirrors the scalar gate
    ``test_multiplicative_miyasawa.py::TestAdditiveSureSelfCheck``).

    RANGE-AGNOSTIC (do not "migrate" to [0,1]): the ``U[-0.5, 0.5]`` draws below
    are a ZERO-MEAN linear-toy signal, not denoiser pixels. Nothing here touches
    ``DOMAIN_MIN``/``DOMAIN_MAX`` or any clip; the analytic reference is built on
    ``E[x^2] = 1/12``, which is a property of that specific zero-mean uniform. A
    literal ``-0.5 -> 0.0`` swap would silently falsify the closed form.
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
# Step 4 (deps: 3): conformal-interval tests. The coverage test uses an IDENTITY
# "denoiser" so the nonconformity score s = |model(noisy) - clean| = |eps| with
# eps ~ N(0, sigma^2) drawn i.i.d. per pixel. Split conformal MUST then attain its
# finite-sample guarantee by construction (exchangeable residuals per sigma bin),
# with NO dependence on any trained checkpoint. Calibration and test splits are
# independent fresh draws so there is zero leakage.
# ---------------------------------------------------------------------------


def _identity_model(x, training=False):
    """A trivial exchangeable "denoiser": returns its input unchanged.

    With ``model(y) = y`` and ``y = clean + eps``, the nonconformity score is
    ``s = |model(y) - clean| = |eps|``, i.e. i.i.d. per-pixel noise magnitude
    independent of position -> split conformal must hit its coverage guarantee by
    construction. Signature matches the ``model(batch, training=False)`` call in
    ``conformal_denoiser_intervals._predict_mu``.
    """
    return x


class TestConformalConstruction:
    """Construction/import: the 5 public conformal functions exist and are callable."""

    def test_public_api_callable(self):
        for fn in (
            conformal_quantile,
            calibrate_per_sigma,
            predict_intervals,
            evaluate_coverage,
        ):
            assert callable(fn), f"{fn!r} is not callable"

    def test_unwrap_point_estimate_contract(self):
        # Plain tensor passes through; list/tuple -> index-0 (deep-supervision).
        t = tf.constant([[1.0, 2.0]])
        assert _unwrap_point_estimate(t) is t
        assert _unwrap_point_estimate([t, "aux"]) is t
        assert _unwrap_point_estimate((t, "aux")) is t


class TestConformalQuantile:
    """Exact finite-sample order-statistic behaviour of ``conformal_quantile``.

    q must equal ``sorted[ceil((n+1)(1-alpha)) - 1]`` (1-indexed rank ``k``), and
    must return ``+inf`` when ``k = ceil((n+1)(1-alpha)) > n`` (calib set too small
    for this alpha). These are the correctness invariants the coverage guarantee
    rests on; do NOT relax them.
    """

    def test_order_statistic_exact(self):
        # n=9, alpha=0.1 -> k=ceil(10*0.9)=9 -> sorted[8] = 0.9 (largest score).
        scores = np.array([0.5, 0.9, 0.1, 0.7, 0.3, 0.8, 0.2, 0.6, 0.4])
        assert conformal_quantile(scores, alpha=0.1) == 0.9

    def test_order_statistic_interior_rank(self):
        # n=9, alpha=0.25 -> k=ceil(10*0.75)=ceil(7.5)=8 -> sorted[7] = 0.8.
        scores = np.array([0.5, 0.9, 0.1, 0.7, 0.3, 0.8, 0.2, 0.6, 0.4])
        assert conformal_quantile(scores, alpha=0.25) == 0.8

    def test_order_statistic_general_formula(self):
        # Cross-check against the closed form for several (n, alpha) on random data.
        rng = np.random.default_rng(3)
        for n in (11, 19, 37, 50):
            for alpha in (0.05, 0.1, 0.2):
                scores = rng.normal(size=n)
                k = int(np.ceil((n + 1) * (1.0 - alpha)))
                got = conformal_quantile(scores, alpha)
                if k > n:
                    assert got == float("inf")
                else:
                    expected = float(np.sort(scores)[k - 1])
                    assert got == expected, (n, alpha, got, expected)

    def test_small_n_returns_inf(self):
        # n=5, alpha=0.1 -> k=ceil(6*0.9)=ceil(5.4)=6 > 5 -> +inf.
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        assert conformal_quantile(scores, alpha=0.1) == float("inf")
        # n=8, alpha=0.1 -> k=ceil(9*0.9)=ceil(8.1)=9 > 8 -> +inf (boundary).
        assert conformal_quantile(np.arange(8.0), alpha=0.1) == float("inf")
        # n=9, alpha=0.1 -> k=9 <= 9 -> finite (just past the +inf boundary).
        assert np.isfinite(conformal_quantile(np.arange(9.0), alpha=0.1))

    def test_invalid_inputs_raise(self):
        with pytest.raises(ValueError):
            conformal_quantile(np.array([1.0, 2.0]), alpha=0.0)
        with pytest.raises(ValueError):
            conformal_quantile(np.array([1.0, 2.0]), alpha=1.0)
        with pytest.raises(ValueError):
            conformal_quantile(np.array([]), alpha=0.1)


# DECISION plan_2026-07-12_e56909cd/D-006: restructure, do NOT literal-swap.
# Do NOT "simplify" this back to a one-sided `np.max(np.abs(noisy)) < 0.5`-shaped
# test, and do NOT swap its literal to `< 1.0`. The pre-migration form encoded the
# zero-center STRUCTURALLY (in its shape, not in a `-0.5` literal a grep can find);
# on [0,1] it would pass for a negative pixel and fail for every legitimately
# bright one. This two-sided, constant-driven guard is also the ONLY thing in this
# file that can detect a wrong domain: measured live, split-conformal coverage is
# 0.900 under BOTH the [0,1] and the legacy [-0.5,+0.5] clip (the clip fires
# identically in calibration and test, so the guarantee is still attained — on a
# meaningless, clip-distorted interval). Deleting or weakening this guard makes the
# domain error SILENT and the suite green. See decisions.md D-006.
def _strictly_in_domain(x) -> bool:
    """True iff every element is strictly inside the ``[0, 1]`` denoiser domain.

    Two-sided BY CONSTRUCTION, and driven off ``DOMAIN_MIN``/``DOMAIN_MAX`` so it
    tracks the module under test rather than re-stating its bounds.
    """
    return bool(np.min(x) > DOMAIN_MIN and np.max(x) < DOMAIN_MAX)


class TestConformalCoverage:
    """The key sanity test: split conformal attains its coverage guarantee on a
    synthetic exchangeable setup (identity model, i.i.d. Gaussian residuals).

    Because residuals are exchangeable BY CONSTRUCTION and calib/test are disjoint
    fresh draws, empirical test coverage must land within ``[target-0.05,
    target+0.05]``. Every draw stays strictly inside ``[0, 1]`` (clean centered on
    mid-grey 0.5 + small sigma) so the point-estimate clip never fires and cannot
    perturb the residual. The calibrator itself is stateless and domain-blind, so
    only the CLIP couples these draws to the domain — hence the in-bounds guard.
    Tolerances are honest; the Pre-Mortem STOP-IF is coverage outside [0.85, 0.95].
    """

    # Large pixel count -> tight empirical coverage (std ~ sqrt(0.9*0.1/N)).
    B, H, W, C = 200, 16, 16, 1

    def _make_split(self, rng, sigma):
        """Independent (clean, noisy, sigma-labels) draw for one bin, one split."""
        shape = (self.B, self.H, self.W, self.C)
        # Clean is a narrow band around mid-grey (the [0,1] domain's interior),
        # sigma small -> noisy stays strictly inside [0, 1] and the clip never fires.
        clean = rng.uniform(0.45, 0.55, size=shape).astype(np.float32)
        eps = rng.normal(0.0, sigma, size=shape).astype(np.float32)
        noisy = (clean + eps).astype(np.float32)
        sigmas = np.full(self.B, sigma, dtype=np.float64)
        return clean, noisy, sigmas

    def test_single_bin_coverage_and_width(self):
        alpha = 0.1
        target = 1.0 - alpha
        sigma = 0.05
        rng = np.random.default_rng(1234)

        # Disjoint fresh draws: calibrate on one, evaluate on the other.
        clean_cal, noisy_cal, sig_cal = self._make_split(rng, sigma)
        clean_te, noisy_te, _ = self._make_split(rng, sigma)

        # Guard: clipping must not fire, else residual != |eps| (exchangeability).
        assert _strictly_in_domain(noisy_cal) and _strictly_in_domain(noisy_te)

        q_by_sigma = calibrate_per_sigma(
            _identity_model, clean_cal, noisy_cal, sig_cal, alpha=alpha,
        )
        q = q_by_sigma[float(sigma)]
        n_pix = clean_cal.size

        res = evaluate_coverage(_identity_model, clean_te, noisy_te, q)
        cov, width = res["coverage"], res["mean_width"]
        print(
            f"\n[coverage single] sigma={sigma} alpha={alpha} calib_n_pix={n_pix} "
            f"q={q:.5f} test_coverage={cov:.4f} target={target} "
            f"mean_width={width:.5f} (2q={2 * q:.5f})"
        )

        assert np.isfinite(q)
        # mean_width == 2q sanity (scalar radius).
        assert abs(width - 2.0 * q) <= 1e-6
        # Honest tolerance: +/-0.05 around target. Pre-Mortem STOP-IF outside this.
        assert target - 0.05 <= cov <= target + 0.05, (
            f"coverage {cov:.4f} outside [{target - 0.05}, {target + 0.05}] "
            f"(STOP-IF gate)"
        )

    def test_smaller_alpha_widens_and_raises_coverage(self):
        # alpha=0.05 -> larger q -> wider intervals -> higher coverage than 0.1.
        sigma = 0.05
        rng = np.random.default_rng(2024)
        clean_cal, noisy_cal, sig_cal = self._make_split(rng, sigma)
        clean_te, noisy_te, _ = self._make_split(rng, sigma)

        q10 = calibrate_per_sigma(
            _identity_model, clean_cal, noisy_cal, sig_cal, alpha=0.10,
        )[float(sigma)]
        q05 = calibrate_per_sigma(
            _identity_model, clean_cal, noisy_cal, sig_cal, alpha=0.05,
        )[float(sigma)]

        cov10 = evaluate_coverage(_identity_model, clean_te, noisy_te, q10)["coverage"]
        cov05 = evaluate_coverage(_identity_model, clean_te, noisy_te, q05)["coverage"]
        print(
            f"\n[coverage alpha-monotone] q(0.10)={q10:.5f} cov={cov10:.4f} | "
            f"q(0.05)={q05:.5f} cov={cov05:.4f}"
        )

        assert q05 > q10, "smaller alpha must give a larger (wider) radius"
        assert cov05 > cov10, "smaller alpha must give higher coverage"
        assert 0.90 <= cov05 <= 0.99  # ~0.95 target, honest band
        assert 0.85 <= cov10 <= 0.95  # ~0.90 target

    def test_mondrian_per_sigma_bins(self):
        # Two sigma bins: each must self-cover ~0.90, and the noisier bin needs a
        # LARGER q (wider intervals track the higher noise level).
        alpha = 0.1
        target = 1.0 - alpha
        sig_lo, sig_hi = 0.03, 0.08
        rng = np.random.default_rng(777)

        # Calibration set: both bins concatenated (calibrate_per_sigma splits them).
        clo_c, nlo_c, slo_c = self._make_split(rng, sig_lo)
        chi_c, nhi_c, shi_c = self._make_split(rng, sig_hi)
        clean_cal = np.concatenate([clo_c, chi_c], axis=0)
        noisy_cal = np.concatenate([nlo_c, nhi_c], axis=0)
        sig_cal = np.concatenate([slo_c, shi_c], axis=0)

        assert _strictly_in_domain(noisy_cal)  # no clip perturbation (two-sided)

        q_by_sigma = calibrate_per_sigma(
            _identity_model, clean_cal, noisy_cal, sig_cal, alpha=alpha,
        )
        q_lo = q_by_sigma[float(sig_lo)]
        q_hi = q_by_sigma[float(sig_hi)]

        # Independent test draws per bin.
        clo_t, nlo_t, _ = self._make_split(rng, sig_lo)
        chi_t, nhi_t, _ = self._make_split(rng, sig_hi)
        cov_lo = evaluate_coverage(_identity_model, clo_t, nlo_t, q_lo)["coverage"]
        cov_hi = evaluate_coverage(_identity_model, chi_t, nhi_t, q_hi)["coverage"]
        print(
            f"\n[mondrian] sig_lo={sig_lo} q={q_lo:.5f} cov={cov_lo:.4f} | "
            f"sig_hi={sig_hi} q={q_hi:.5f} cov={cov_hi:.4f} target={target}"
        )

        # Noisier bin -> larger radius.
        assert q_hi > q_lo, f"higher sigma must yield larger q ({q_hi} !> {q_lo})"
        # Each bin self-covers within the honest band (STOP-IF outside [0.85,0.95]).
        for name, cov in (("lo", cov_lo), ("hi", cov_hi)):
            assert target - 0.05 <= cov <= target + 0.05, (
                f"bin {name} coverage {cov:.4f} outside "
                f"[{target - 0.05}, {target + 0.05}] (STOP-IF gate)"
            )
