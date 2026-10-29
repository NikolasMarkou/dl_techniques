"""
Sequence-model arm of the Fisher tool, on a tiny causal model whose sequence space can be
enumerated exactly (V = 4 tokens, prompt of 2, response of 2: 16 responses).

The enumeration gives closed forms for everything the paper claims:

* the martingale identity (Prop. 2): E[sum_t xi_t^2] equals E[h^2] exactly, although the
  cross terms xi_u xi_t are NOT zero for any single sequence;
* the mean-token estimator equals F / T at fixed response length T (it averages over
  positions, where the sequence Fisher sums over them), which pins both the pair scale
  1/sqrt(2) (a 1/2 scale would give half of it) and the rule that a == b pairs stay in the
  count (dropping them would bias it upward);
* the Uniform reference is the variance of the logit sensitivities under the uniform
  distribution over the full vocabulary.
"""

import itertools

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.analyzer import fisher_metrics as fm
from dl_techniques.analyzer.fisher_metrics import (
    FisherConfig,
    FisherSignal,
    canonical_layout,
    compute_diagonal_fisher,
    compute_token_fisher,
)
from tests.numerics import true_float32_matmul

V = 4
PROMPT_LEN = 2
T = 2


@pytest.fixture(autouse=True)
def cpu_float32():
    with keras.device("cpu"), true_float32_matmul():
        yield


def make_lm(seed: int = 0, scale: float = 1.0) -> keras.Model:
    """A causal token model: running mean of embeddings, then an MLP to V logits."""
    keras.utils.set_random_seed(seed)
    ids = keras.layers.Input((None,), dtype="int32")
    emb = keras.layers.Embedding(V, 6)(ids)
    ctx = keras.layers.Lambda(
        lambda e: keras.ops.cumsum(e, axis=1)
        / keras.ops.cast(keras.ops.arange(1, keras.ops.shape(e)[1] + 1), "float32")[None, :, None])(emb)
    hidden = keras.layers.Dense(8, activation="tanh")(ctx)
    model = keras.Model(ids, keras.layers.Dense(V)(hidden))
    if scale != 1.0:
        model.set_weights([w * scale for w in model.get_weights()])
    return model


PROMPTS = np.array([[0, 1], [2, 3], [1, 1], [3, 0]], dtype="int32")


def cfg(**kw) -> FisherConfig:
    base = dict(batch_size=8, max_tokens=T, token_level=True, n_positions_per_response=T,
                n_token_pairs_per_position=6)
    base.update(kw)
    return FisherConfig(**base)


def flat(grads):
    # An embedding lookup yields IndexedSlices; densify before flattening.
    return np.concatenate([np.asarray(tf.convert_to_tensor(g)).reshape(-1) for g in grads]).astype(np.float64)


def token_log_probs(model, row):
    """log p(row[PROMPT_LEN + t] | row[: PROMPT_LEN + t]) for t < T, differentiable."""
    logits = model(row[None], training=False)[0]
    logp = keras.ops.log_softmax(logits, axis=-1)
    return [logp[PROMPT_LEN + t - 1, row[PROMPT_LEN + t]] for t in range(T)]


def enumerate_sequences(model, prompt):
    """All V^T responses with their probability, per-token scores xi_t and sequence score h."""
    rows = []
    for response in itertools.product(range(V), repeat=T):
        row = np.concatenate([prompt, response]).astype("int32")
        with tf.GradientTape(persistent=True) as tape:
            terms = token_log_probs(model, row)
            total = terms[0] + terms[1]
        xi = [flat(tape.gradient(t, model.trainable_variables, unconnected_gradients="zero")) for t in terms]
        h = flat(tape.gradient(total, model.trainable_variables, unconnected_gradients="zero"))
        prob = float(np.exp(sum(float(t) for t in terms)))
        rows.append((prob, xi, h))
    return rows


def sensitivities(model, prefix):
    """j[a, i] = d z_a / d theta_i at the last position of ``prefix``, shape (V, P)."""
    out = []
    for a in range(V):
        with tf.GradientTape() as tape:
            z = model(prefix[None], training=False)[0, -1, a]
        out.append(flat(tape.gradient(z, model.trainable_variables, unconnected_gradients="zero")))
    return np.stack(out)


class TestTheIdentities:

    def test_cross_terms_vanish_in_expectation_but_not_per_sequence(self):
        model = make_lm()
        prompt = PROMPTS[0]
        rows = enumerate_sequences(model, prompt)
        seq_fisher = sum(p * h ** 2 for p, _, h in rows)
        token_sum_fisher = sum(p * (xi[0] ** 2 + xi[1] ** 2) for p, xi, _ in rows)
        cross = [2 * xi[0] * xi[1] for _, xi, _ in rows]
        expected_cross = sum(p * c for (p, _, _), c in zip(rows, cross))
        np.testing.assert_allclose(seq_fisher, token_sum_fisher, rtol=1e-4, atol=1e-9)
        assert np.abs(expected_cross).max() < 1e-6
        # the naive per-sample identity h^2 = sum_t xi_t^2 is FALSE: a single sequence's
        # cross term is far from zero
        assert max(np.abs(c).max() for c in cross) > 1e-3
        per_sample_gap = max(np.abs(h ** 2 - (xi[0] ** 2 + xi[1] ** 2)).max() for _, xi, h in rows)
        assert per_sample_gap > 1e-3

    def test_token_fisher_is_the_variance_of_the_logit_sensitivities(self):
        """f_i(s) = sum_a p_a (j_ia - jbar_i)^2 = 1/2 E_{a,b iid p}[(j_ia - j_ib)^2]."""
        model = make_lm()
        prefix = np.concatenate([PROMPTS[1], [2]]).astype("int32")
        probs = np.asarray(keras.ops.softmax(model(prefix[None])[0, -1], axis=-1), np.float64)
        J = sensitivities(model, prefix)
        mean = probs @ J
        variance = probs @ (J - mean) ** 2
        pairwise = 0.5 * sum(probs[a] * probs[b] * (J[a] - J[b]) ** 2 for a in range(V) for b in range(V))
        np.testing.assert_allclose(variance, pairwise, rtol=1e-6, atol=1e-12)
        # and it equals E_a[(d log p_a)^2], the definition
        direct = 0.0
        for a in range(V):
            with tf.GradientTape() as tape:
                lp = keras.ops.log_softmax(model(prefix[None])[0, -1], axis=-1)[a]
            direct = direct + probs[a] * flat(tape.gradient(lp, model.trainable_variables)) ** 2
        np.testing.assert_allclose(variance, direct, rtol=1e-4, atol=1e-9)


class TestTheSequenceArm:

    def test_sampled_sequences_follow_the_models_own_distribution(self):
        model = make_lm()
        prompt = PROMPTS[0]
        rows = enumerate_sequences(model, prompt)
        exact = np.array([p for p, _, _ in rows])
        scorer = fm._Scorer(model, fm.KIND_SEQUENCE, cfg(), None, None)
        n = 6000
        responses = scorer.sample(np.tile(prompt, (n, 1)), np.random.default_rng(0))
        index = {r: i for i, r in enumerate(itertools.product(range(V), repeat=T))}
        counts = np.zeros(len(exact))
        for r in responses:
            counts[index[tuple(int(t) for t in r)]] += 1
        assert np.abs(counts / n - exact).max() < 0.03
        assert exact.max() < 0.5  # anti-vacuity: not a degenerate point mass

    def test_a_stopping_token_ends_a_response_and_is_kept_but_truncation_adds_none(self):
        model = make_lm(scale=0.2)
        signal = FisherSignal(eos_token_id=0)
        scorer = fm._Scorer(model, fm.KIND_SEQUENCE, cfg(max_tokens=6), signal, None)
        responses = scorer.sample(np.tile(PROMPTS[0], (300, 1)), np.random.default_rng(0))
        for r in responses:
            assert 1 <= len(r) <= 6
            assert 0 not in r[:-1]                       # nothing after the first stop token
            assert r[-1] == 0 or len(r) == 6             # stopped, or truncated whole
        assert any(r[-1] == 0 and len(r) < 6 for r in responses)
        assert any(len(r) == 6 and r[-1] != 0 for r in responses)  # truncation exists, no synthetic EOS

    def test_sequence_fisher_matches_the_enumerated_fisher(self):
        model = make_lm()
        exact = sum(p * h ** 2 for p, _, h in enumerate_sequences(model, PROMPTS[0]))
        out = compute_diagonal_fisher(model, PROMPTS, cfg(), np.random.default_rng(0),
                                      indices=[0] * 1200)
        assert out.output_kind == fm.KIND_SEQUENCE
        z = np.abs(out.fisher_flat - exact) / np.maximum(out.standard_errors, 1e-12)
        assert np.mean(z <= 4.5) >= 0.97
        assert exact.max() > 1e-3

    def test_prompt_scope_adds_exactly_the_prompt_terms(self):
        model = make_lm()
        row = np.concatenate([PROMPTS[1], [3, 1]]).astype("int32")
        resp = fm._Scorer(model, fm.KIND_SEQUENCE, cfg(), None, canonical_layout(model))
        both = fm._Scorer(model, fm.KIND_SEQUENCE, cfg(score_scope="response_plus_prompt"), None,
                          canonical_layout(model))
        lp_resp, terms_resp, _ = resp.score(PROMPTS[1][None], row[PROMPT_LEN:], np.float64)
        lp_both, terms_both, _ = both.score(PROMPTS[1][None], row[PROMPT_LEN:], np.float64)
        logits = np.asarray(model(row[None]))[0]
        logp = logits - np.log(np.exp(logits).sum(-1, keepdims=True))
        prompt_term = logp[0, row[1]]  # position 0 predicts prompt token 1
        assert len(terms_resp) == T and len(terms_both) == T + 1
        assert lp_both == pytest.approx(lp_resp + prompt_term, abs=1e-5)

    def test_an_ambiguous_rank3_output_names_the_override(self):
        keras.utils.set_random_seed(0)
        model = keras.Sequential([keras.layers.Input((4, 3)), keras.layers.Dense(5)])
        x = np.random.default_rng(0).normal(size=(4, 4, 3)).astype("float32")
        with pytest.raises(ValueError, match="fisher_force_sequence"):
            fm.infer_output_kind(model, x, FisherConfig())

    def test_a_multi_output_model_demands_a_joint_likelihood(self):
        inp = keras.layers.Input((3,))
        model = keras.Model(inp, [keras.layers.Dense(2)(inp), keras.layers.Dense(2)(inp)])
        with pytest.raises(ValueError, match="output_logprob_fn"):
            fm.infer_output_kind(model, np.zeros((2, 3), "float32"), FisherConfig())

    def test_a_custom_likelihood_without_a_sampler_is_refused(self):
        model = make_lm()
        with pytest.raises(ValueError, match="sampler"):
            fm.infer_output_kind(model, PROMPTS, FisherConfig(),
                                 FisherSignal(output_logprob_fn=lambda m, x, y: 0.0))


class TestTheTokenArm:

    @pytest.fixture(scope="class")
    def measured(self):
        with keras.device("cpu"), true_float32_matmul():
            model = make_lm()
            exact_f = sum(p * h ** 2 for p, _, h in enumerate_sequences(model, PROMPTS[0]))
            exact_unif = np.zeros_like(exact_f)
            # E over the model's own prefixes of 1/T sum_t Var_U[j(s_t)]
            for response in itertools.product(range(V), repeat=T):
                row = np.concatenate([PROMPTS[0], response]).astype("int32")
                prob = float(np.exp(sum(float(t) for t in token_log_probs(model, row))))
                for t in range(T):
                    J = sensitivities(model, row[:PROMPT_LEN + t])
                    exact_unif += prob * J.var(axis=0) / T
            fcfg = cfg(n_token_pairs_per_position=8)
            out = compute_diagonal_fisher(model, PROMPTS, fcfg, np.random.default_rng(0), indices=[0] * 260)
            tok = compute_token_fisher(model, PROMPTS, out, fcfg, np.random.default_rng(1))
            return model, exact_f, exact_unif, out, tok

    @staticmethod
    def by_layer(model, vec):
        layout = canonical_layout(model)
        names, spans = fm.layer_groups(layout)
        return names, np.array([sum(vec[s:e].sum() for s, e in sp) for sp in spans])

    def test_mean_token_fisher_is_the_sequence_fisher_over_the_response_length(self, measured):
        """T = 2 fixed, so the mean over positions of f is F / 2. A 1/2 pair scale would give F / 4."""
        model, exact_f, _, _, tok = measured
        names, got = self.by_layer(model, tok.token_fisher_mean)
        _, want = self.by_layer(model, exact_f / T)
        np.testing.assert_allclose(got, want, rtol=0.2)
        assert tok.estimator == fm.ESTIMATOR_SEQUENCE  # the sequence field is untouched by the token arm
        assert tok.token_fisher_mean is not tok.fisher_flat

    def test_the_uniform_reference_is_the_variance_under_uniform_pairs(self, measured):
        model, _, exact_unif, _, tok = measured
        _, got = self.by_layer(model, tok.uniform_reference)
        _, want = self.by_layer(model, exact_unif)
        np.testing.assert_allclose(got, want, rtol=0.2)

    def test_group_matrices_and_ratios_are_published_per_parameter_group(self, measured):
        model, _, _, _, tok = measured
        assert tok.group_names and len(tok.group_names) == len(set(tok.group_names))
        assert tok.token_fisher_by_position.shape == (T, len(tok.group_names))
        assert tok.uniform_by_position.shape == tok.token_fisher_by_position.shape
        assert set(tok.real_uniform_ratio) == set(tok.group_names)
        assert fm.FM.REAL_UNIFORM_RATIO in tok.metrics

    def test_the_token_arm_refuses_a_non_sequence_output(self):
        keras.utils.set_random_seed(0)
        clf = keras.Sequential([keras.layers.Input((3,)), keras.layers.Dense(2)])
        out = compute_diagonal_fisher(clf, np.zeros((4, 3), "float32") + 0.1,
                                      FisherConfig(n_samples=4, assert_no_weight_drift=False),
                                      np.random.default_rng(0))
        with pytest.raises(ValueError, match="sequence"):
            compute_token_fisher(clf, np.zeros((4, 3), "float32"), out, FisherConfig(), np.random.default_rng(0))
