"""Fast CPU pytest for the multiplicative-noise Miyasawa compliance tooling.

Hard compliance gates:
  * relation (A) rel-RMSE <= 0.06 at sigma=0.15 on the signed mixture prior, and beats
    the D(y)=y baseline.
  * relation (B) rel-RMSE <= 0.06 at sigma=0.15, and beats the baseline.
  * the additive-SURE self-check (Pre-Mortem STOP-IF): the Hutchinson divergence
    estimator reproduces the additive closed-form SURE risk on a linear toy denoiser
    within MC tolerance. If this gate could not be met it would be demoted (see module
    docstring / decisions.md D-002); as authored it passes and stays a hard gate.
  * apply_multiplicative_gaussian sample mean/variance match x and (sigma*x)^2, and the
    primitive is traceable inside tf.data.Dataset.map.

All RNG seeds are fixed and n_samples is scaled down from the 8M reference to keep the
whole suite well under 60s on CPU.
"""

import numpy as np
import tensorflow as tf

from dl_techniques.utils.multiplicative_miyasawa import (
    apply_multiplicative_gaussian,
    mc_posterior_mean,
    relation_A,
    relation_B,
    rel_rmse,
    additive_sure_risk,
    hutchinson_divergence,
    sure_divergence_consistency,
    signed_mixture_prior,
)

SIGMA = 0.15
# Scaled down from the 8M / 600-bin reference. Fewer bins -> more samples/bin -> a
# lower np.gradient noise floor, so 3M samples at 300 bins reaches rel-RMSE ~0.03/0.04
# (better than the 8M/600 reference's 0.04/0.05) in <1s, with wide headroom under 0.06.
N_SAMPLES = 3_000_000
N_BINS = 300
TOL = 0.06
BASELINE = 0.080  # reference D(y)=y rel-RMSE


def _build_mc():
    # Draw the prior directly at the MC sample count (continuous draw, matching the
    # reference) so binning is not limited by a small discrete prior pool.
    prior = signed_mixture_prior(N_SAMPLES, seed=0)
    return mc_posterior_mean(prior, sigma=SIGMA, n_samples=N_SAMPLES, n_bins=N_BINS, seed=0)


class TestRelations:
    def test_relation_A(self):
        mc = _build_mc()
        rhs = relation_A(mc)
        err = rel_rmse(rhs, mc["Ex"], mc["mask"])
        base = rel_rmse(mc["ctr"], mc["Ex"], mc["mask"])
        print(f"\n[relation A] rel-RMSE={err:.4f}  baseline(D=y)={base:.4f}")
        assert err <= TOL, f"relation (A) rel-RMSE {err:.4f} > {TOL}"
        assert err < base, f"relation (A) {err:.4f} does not beat baseline {base:.4f}"

    def test_relation_B(self):
        mc = _build_mc()
        rhs = relation_B(mc)
        err = rel_rmse(rhs, mc["Ex"], mc["mask"])
        base = rel_rmse(mc["ctr"], mc["Ex"], mc["mask"])
        print(f"\n[relation B] rel-RMSE={err:.4f}  baseline(D=y)={base:.4f}")
        assert err <= TOL, f"relation (B) rel-RMSE {err:.4f} > {TOL}"
        assert err < base, f"relation (B) {err:.4f} does not beat baseline {base:.4f}"


class TestAdditiveSureSelfCheck:
    """Pre-Mortem STOP-IF: validate the divergence estimator on the additive closed form."""

    def test_hutchinson_divergence_linear(self):
        # Linear toy D(y) = a*y has exact divergence a*N.
        a = 0.7
        tf.random.set_seed(0)
        y = tf.random.normal([4, 8, 8, 2], dtype=tf.float32)
        n = float(tf.size(y).numpy())
        div = hutchinson_divergence(lambda t: a * t, y, n_hutchinson=16, eps=1e-3, seed=0)
        print(f"\n[hutchinson div] est={div:.2f} exact={a * n:.2f} (N={n:.0f})")
        # Finite-difference of an exactly linear map is exact up to fp; tight tol.
        assert abs(div - a * n) <= 0.02 * a * n

    def test_additive_sure_matches_true_mse(self):
        # Toy: x ~ N(0,1), y = x + sigma*eps. Linear denoiser D(y)=a*y.
        # True MSE = E||a*y - x||^2 = sum_i [ (a-1)^2 x_i^2 + a^2 sigma^2 eps_i^2 ]
        # (cross term zero in expectation; here we compute the realized values so SURE,
        # which is unbiased per-realization only in expectation, is compared on the
        # *empirical* risk averaged over many elements -> matches within MC tol).
        sigma = 0.3
        a = 0.6
        rng = np.random.default_rng(0)
        N = 4 * 32 * 32 * 3
        x = rng.normal(0.0, 1.0, N).astype(np.float32)
        eps = rng.normal(0.0, 1.0, N).astype(np.float32)
        y_np = x + sigma * eps
        y = tf.reshape(tf.constant(y_np), [4, 32, 32, 3])

        denoiser = lambda t: a * t
        d = a * y_np
        true_mse = float(np.sum((d - x) ** 2))

        res = additive_sure_risk(denoiser, y, sigma=sigma, n_hutchinson=16, eps=1e-3, seed=0)
        sure = res["sure_risk"]
        rel = abs(sure - true_mse) / true_mse
        print(f"\n[additive SURE] sure={sure:.1f} true_mse={true_mse:.1f} rel={rel:.4f}")
        # div is exact for linear; remaining gap is the SURE-vs-realized-risk MC noise.
        assert rel <= 0.10, f"additive SURE self-check rel error {rel:.4f} > 0.10"


class TestSureOnTinyModel:
    def test_sure_returns_finite_on_tiny_denoiser(self):
        from dl_techniques.models.bias_free_denoisers.bfconvunext import (
            create_convunext_denoiser,
        )

        model = create_convunext_denoiser(
            input_shape=(16, 16, 3),
            depth=3,  # model enforces depth >= 3
            initial_filters=4,
            blocks_per_level=1,
            drop_path_rate=0.0,
        )
        tf.random.set_seed(0)
        clean = tf.random.uniform([1, 16, 16, 3], minval=-1.0, maxval=1.0)
        noisy = apply_multiplicative_gaussian(clean, SIGMA)

        out = sure_divergence_consistency(
            model, noisy, sigma=SIGMA, n_hutchinson=4, eps=1e-3, seed=0
        )
        for k, v in out.items():
            assert np.isfinite(v), f"{k} is not finite: {v}"
        print(f"\n[tiny SURE] {out}")


class TestNoisePrimitive:
    def test_mean_and_variance(self):
        tf.random.set_seed(0)
        for x_val in (-0.5, 0.6):
            x = tf.fill([400_000], float(x_val))
            y = apply_multiplicative_gaussian(x, SIGMA).numpy()
            mean = float(np.mean(y))
            var = float(np.var(y))
            target_var = (SIGMA * x_val) ** 2
            print(f"\n[noise] x={x_val} mean={mean:.4f} var={var:.5f} target_var={target_var:.5f}")
            assert abs(mean - x_val) < 0.01
            assert abs(var - target_var) < 0.05 * target_var + 1e-6

    def test_graph_traceable_in_tf_data(self):
        clean = tf.random.uniform([8, 4, 4, 1], minval=-1.0, maxval=1.0)
        ds = tf.data.Dataset.from_tensor_slices(clean)
        sigma_var = tf.Variable(SIGMA, dtype=tf.float32)
        ds = ds.map(lambda t: apply_multiplicative_gaussian(t, sigma_var))
        got = list(ds.as_numpy_iterator())
        assert len(got) == 8
        assert all(np.all(np.isfinite(g)) for g in got)
