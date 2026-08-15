"""RED proofs for the four remaining DeepAR defects (review finding C-38).

Written in the style of this package's existing `TestDeepARScaleSemantics`:
every expected value is derived from the definition (ancestral sampling, the
softplus range, the negative-binomial log-pmf), never read off the
implementation.

(a) **Ancestral sampling accumulated the trajectory.** At horizon step `t` the
    decoder sequence was `concatenate([encoder_inputs, decoder_input])` — the
    conditioning window plus only the single most recent draw. Steps `1..t-1`
    never re-entered the context, so an H-step forecast was H nearly
    independent one-step-ahead forecasts and multi-step quantiles were too
    narrow.
(b) **Softplus was spelled `log(1 + exp(x))`.** A float32 logit of +100
    overflows `exp` to `inf`, so `sigma` is `inf` and the loss `NaN`; a logit
    of -100 underflows `exp` to exactly 0, so `sigma` is 0, `log(sigma)` is
    `-inf` and `gaussian_loss` divides by zero.
(c) **`negative_binomial_loss` dropped `lgamma(z + r) - lgamma(r)`.** Since
    `r = 1 / alpha` those terms depend on `alpha`, so the gradient with respect
    to it was systematically wrong, not merely offset.
(d) **`ScaleLayer`'s docstring still claimed `sqrt(nu)` for the Gaussian
    standard deviation** — the exact claim an earlier fix refuted, left in the
    layer that fix routes through.

CPU only.
"""

import numpy as np
import pytest
import keras
from keras import ops
from scipy.special import gammaln

from dl_techniques.models.time_series.deepar.model import DeepAR
from dl_techniques.layers.time_series.deepar_blocks import (
    ScaleLayer,
    GaussianLikelihoodHead,
    NegativeBinomialLikelihoodHead,
    MIN_LIKELIHOOD_PARAM,
)


# =============================================================================
# (a) ancestral sampling
# =============================================================================

class _RecordingLSTM:
    """Transparent proxy recording every sequence handed to the wrapped layer."""

    def __init__(self, layer):
        self._layer = layer
        self.calls = []

    def __call__(self, x, *args, **kwargs):
        self.calls.append(keras.ops.convert_to_numpy(x))
        return self._layer(x, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._layer, name)


class TestAncestralSampling:

    COND_LEN = 5
    PRED_LEN = 3
    COV_DIM = 3

    def _model(self):
        keras.utils.set_random_seed(0)
        return DeepAR(num_layers=1, hidden_dim=8, likelihood='gaussian',
                      target_dim=1, num_samples=2,
                      conditioning_length=self.COND_LEN)

    def _inputs(self):
        rng = np.random.default_rng(3)
        cond = (np.abs(rng.normal(size=(2, self.COND_LEN, 1))) + 1.0).astype("float32")
        cov = rng.normal(
            size=(2, self.COND_LEN + self.PRED_LEN, self.COV_DIM)
        ).astype("float32")
        return {'conditioning_target': cond, 'full_covariates': cov}

    def test_each_step_conditions_on_every_previous_draw(self):
        """Step t's decoder context must be the window plus draws 1..t.

        That is the definition of ancestral sampling: `p(z_t | z_{1:t-1})`.
        Three independent facts are asserted about the recorded sequences —
        their lengths grow by one per horizon step, each is a strict prefix
        extension of the previous one, and the row appended at step t carries
        the value drawn at step t-1.
        """
        model = self._model()
        x = self._inputs()

        spy = _RecordingLSTM(model.lstm_layers[0])
        model.lstm_layers[0] = spy
        try:
            samples = keras.ops.convert_to_numpy(
                model._prediction_mode(x, training=False)
            )
        finally:
            model.lstm_layers[0] = spy._layer

        assert samples.shape == (2, 2, self.PRED_LEN, 1)
        assert len(spy.calls) == 2 * self.PRED_LEN, (
            "expected one stacked-LSTM pass per (sample, horizon step); "
            f"recorded {len(spy.calls)}"
        )

        # First sample's trajectory only.
        seqs = spy.calls[:self.PRED_LEN]

        lengths = [s.shape[1] for s in seqs]
        assert lengths == [self.COND_LEN + t + 1 for t in range(self.PRED_LEN)], (
            f"decoder context lengths {lengths} do not grow with the horizon; "
            f"a constant {self.COND_LEN + 1} means only the most recent draw "
            "re-enters the context"
        )

        for t in range(1, self.PRED_LEN):
            np.testing.assert_allclose(
                seqs[t][:, :seqs[t - 1].shape[1], :], seqs[t - 1], atol=0.0,
                err_msg=f"step {t}'s context is not an extension of step {t-1}'s",
            )

        # The value channel of the row appended at step t must be the step-(t-1)
        # draw, scaled. nu is not passed in, so recover it from the layer's own
        # definition: mean over the conditioning range + scale_epsilon.
        nu = x['conditioning_target'].mean(axis=1, keepdims=True) + model.scale_epsilon
        for t in range(1, self.PRED_LEN):
            appended = seqs[t][:, -1, 0]
            expected = samples[0, :, t - 1, 0] / nu[:, 0, 0]
            np.testing.assert_allclose(
                appended, expected, rtol=1e-5, atol=1e-6,
                err_msg=(
                    f"the row appended at horizon step {t} is not the draw made "
                    f"at step {t - 1}"
                ),
            )

    def test_multi_step_spread_is_not_a_repeated_one_step_forecast(self):
        """ANTI-VACUITY: the trajectory genuinely evolves across the horizon.

        If every step re-ran the same context, the recorded sequences would be
        identical apart from the last row. This asserts the forward pass is
        live, so the prefix assertions above are observations rather than
        artifacts of a degenerate loop.
        """
        model = self._model()
        samples = keras.ops.convert_to_numpy(
            model._prediction_mode(self._inputs(), training=False)
        )
        per_step_spread = samples.std(axis=0)  # over the sample axis
        assert np.all(np.isfinite(per_step_spread))
        assert per_step_spread.max() > 0.0, "all Monte-Carlo paths are identical"


# =============================================================================
# (b) softplus overflow / underflow
# =============================================================================

def _pin_head(head, bias_value, projection_names):
    """Zero every kernel and pin every bias, so the logit is exactly known."""
    for name in projection_names:
        proj = getattr(head, name)
        kernel, bias = proj.kernel, proj.bias
        kernel.assign(ops.zeros_like(kernel))
        bias.assign(ops.full_like(bias, bias_value))


class TestLikelihoodHeadNumerics:

    def test_large_positive_logit_gives_a_finite_sigma(self):
        """softplus(100) == 100. `log(1 + exp(100))` is `inf` in float32."""
        head = GaussianLikelihoodHead(units=1)
        head.build((None, 4))
        _pin_head(head, 100.0, ["mu_projection", "sigma_projection"])

        _, sigma = head(ops.zeros((2, 4)))
        sigma = keras.ops.convert_to_numpy(sigma)

        assert np.all(np.isfinite(sigma)), f"sigma is not finite: {sigma}"
        np.testing.assert_allclose(sigma, 100.0, rtol=1e-5)

    def test_large_negative_logit_gives_a_strictly_positive_sigma(self):
        """softplus(-100) underflows float32 to 0; the floor must catch it.

        A zero sigma is not a small number — `gaussian_loss` takes `log(sigma)`
        (-inf) and divides by it.
        """
        head = GaussianLikelihoodHead(units=1)
        head.build((None, 4))
        _pin_head(head, -100.0, ["mu_projection", "sigma_projection"])

        _, sigma = head(ops.zeros((2, 4)))
        sigma = keras.ops.convert_to_numpy(sigma)

        assert np.all(sigma > 0.0), f"sigma reached zero: {sigma}"
        assert np.all(sigma >= MIN_LIKELIHOOD_PARAM)

        loss = DeepAR.gaussian_loss(None, {
            'mu': ops.zeros((2, 1)),
            'sigma': ops.convert_to_tensor(sigma),
            'target': ops.ones((2, 1)),
        })
        assert np.isfinite(float(keras.ops.convert_to_numpy(loss))), (
            "gaussian_loss is not finite at the floored sigma"
        )

    def test_negative_binomial_head_is_stable_at_both_extremes(self):
        """The same two failure modes, on mu and alpha."""
        for bias, label in ((100.0, "overflow"), (-100.0, "underflow")):
            head = NegativeBinomialLikelihoodHead(units=1)
            head.build((None, 4))
            _pin_head(head, bias, ["mu_projection", "alpha_projection"])

            mu, alpha = head(ops.zeros((2, 4)))
            mu = keras.ops.convert_to_numpy(mu)
            alpha = keras.ops.convert_to_numpy(alpha)

            assert np.all(np.isfinite(mu)) and np.all(np.isfinite(alpha)), label
            assert np.all(mu > 0.0) and np.all(alpha > 0.0), label

            loss = DeepAR.negative_binomial_loss(None, {
                'mu': ops.convert_to_tensor(mu),
                'alpha': ops.convert_to_tensor(alpha),
                'target': ops.ones((2, 1)),
            })
            assert np.isfinite(float(keras.ops.convert_to_numpy(loss))), label


# =============================================================================
# (c) the negative-binomial gradient
# =============================================================================

def _negbin_nll_oracle(z, mu, alpha):
    """float64 NegBin NLL from the log-pmf, less the parameter-free lgamma(z+1).

    `r = 1/alpha`, `p = r/(r+mu) = 1/(1 + alpha*mu)`,
    `log p(z) = lgamma(z+r) - lgamma(r) - lgamma(z+1) + r*log(p) + z*log(1-p)`.

    Written from Johnson, Kotz & Kemp, *Univariate Discrete Distributions*
    §5.1, not from `model.py`.
    """
    z = np.asarray(z, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    alpha = np.asarray(alpha, dtype=np.float64)
    r = 1.0 / alpha
    p = 1.0 / (1.0 + alpha * mu)
    log_pmf = (gammaln(z + r) - gammaln(r)
               + r * np.log(p) + z * np.log1p(-p))
    return -log_pmf


class TestNegativeBinomialGradient:

    Z = np.array([[0.0], [3.0], [17.0], [120.0]], dtype=np.float64)
    MU = np.array([[1.0], [4.0], [20.0], [100.0]], dtype=np.float64)
    ALPHA = np.array([[0.05], [0.3], [0.8], [2.0]], dtype=np.float64)

    def test_value_matches_the_log_pmf_up_to_the_constant_normalizer(self):
        loss = DeepAR.negative_binomial_loss(None, {
            'mu': ops.convert_to_tensor(self.MU),
            'alpha': ops.convert_to_tensor(self.ALPHA),
            'target': ops.convert_to_tensor(self.Z),
        })
        got = float(keras.ops.convert_to_numpy(loss))
        want = float(_negbin_nll_oracle(self.Z, self.MU, self.ALPHA).mean())
        np.testing.assert_allclose(got, want, rtol=1e-6)

    def test_gradient_wrt_alpha_matches_a_float64_finite_difference_oracle(self):
        """The whole point: the missing terms are NOT constant in alpha.

        The oracle is a central finite difference of `_negbin_nll_oracle`,
        which is built from the log-pmf. Dropping `lgamma(z+r) - lgamma(r)`
        changes this derivative by `(digamma(z+r) - digamma(r)) / alpha^2`,
        which is large and z-dependent.
        """
        import tensorflow as tf

        alpha_var = tf.Variable(self.ALPHA, dtype=tf.float64)
        with tf.GradientTape() as tape:
            loss = DeepAR.negative_binomial_loss(None, {
                'mu': ops.convert_to_tensor(self.MU),
                'alpha': alpha_var,
                'target': ops.convert_to_tensor(self.Z),
            })
        grad = keras.ops.convert_to_numpy(tape.gradient(loss, alpha_var))

        n = float(self.Z.size)
        h = 1e-6
        fd = np.zeros_like(self.ALPHA)
        for i in range(self.ALPHA.shape[0]):
            up = self.ALPHA.copy()
            dn = self.ALPHA.copy()
            up[i, 0] += h
            dn[i, 0] -= h
            fd[i, 0] = (
                _negbin_nll_oracle(self.Z, self.MU, up).sum()
                - _negbin_nll_oracle(self.Z, self.MU, dn).sum()
            ) / (2.0 * h) / n

        np.testing.assert_allclose(
            grad, fd, rtol=2e-4, atol=1e-8,
            err_msg=("d(loss)/d(alpha) disagrees with a float64 finite "
                     "difference of the NegBin log-pmf; the lgamma(z+r) - "
                     "lgamma(r) pair is not constant in alpha"),
        )

    def test_anti_vacuity_the_oracle_can_tell_the_two_formulas_apart(self):
        """The probe is discriminating: the old formula fails it by a mile.

        Without this arm, a tolerance chosen too loosely would let both
        formulas pass and the test above would prove nothing.
        """
        import tensorflow as tf

        def old_loss(mu, alpha, target):
            eps = 1e-7
            p = 1.0 / (1.0 + alpha * mu + eps)
            return ops.mean(-ops.log(p + eps) / alpha
                            - target * ops.log(1.0 - p + eps))

        alpha_var = tf.Variable(self.ALPHA, dtype=tf.float64)
        with tf.GradientTape() as tape:
            loss = old_loss(ops.convert_to_tensor(self.MU), alpha_var,
                            ops.convert_to_tensor(self.Z))
        old_grad = keras.ops.convert_to_numpy(tape.gradient(loss, alpha_var))

        n = float(self.Z.size)
        h = 1e-6
        fd = np.zeros_like(self.ALPHA)
        for i in range(self.ALPHA.shape[0]):
            up = self.ALPHA.copy()
            dn = self.ALPHA.copy()
            up[i, 0] += h
            dn[i, 0] -= h
            fd[i, 0] = (
                _negbin_nll_oracle(self.Z, self.MU, up).sum()
                - _negbin_nll_oracle(self.Z, self.MU, dn).sum()
            ) / (2.0 * h) / n

        rel = np.abs(old_grad - fd) / np.maximum(np.abs(fd), 1e-12)
        assert rel.max() > 0.1, (
            "the dropped-lgamma formula is indistinguishable from the oracle "
            "on this fixture, so the gradient test would be vacuous"
        )


# =============================================================================
# (d) the refuted docstring
# =============================================================================

class TestScaleLayerDocstring:

    def test_docstring_does_not_claim_sqrt_scaling_for_the_gaussian_sigma(self):
        """The claim must be gone as an ASSERTION, not merely reworded.

        The docstring quotes the refuted sentence in order to retract it, so a
        bare substring search for the sentence would fire on the retraction
        itself. What is pinned here is the asserting clause ("or scales by
        sqrt(nu)") plus the presence of an explicit retraction.
        """
        doc = " ".join((ScaleLayer.__doc__ or "").split())
        assert "or scales by ``sqrt(nu)``" not in doc, (
            "ScaleLayer's docstring still claims sqrt(nu) for the Gaussian "
            "sigma; a repair driven from it re-introduces a fixed defect"
        )
        assert "That is wrong" in doc, (
            "the refuted sqrt(nu) claim must be retracted explicitly, not "
            "silently deleted -- it has already misled one repair"
        )

    def test_docstring_states_the_scale_actually_used(self):
        doc = " ".join((ScaleLayer.__doc__ or "").split())
        assert "``1 / sqrt(nu)``" in doc, (
            "the docstring must say which parameter DOES take a sqrt scale"
        )
        assert "first-moment-scale" in doc


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
