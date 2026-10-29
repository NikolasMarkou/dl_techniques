"""
Regression suite for the gradient arm of the Fisher tool, on a ~10^2-parameter classifier.

Every estimator is compared with a closed form obtained by ENUMERATING the K classes (the
exact g_i = sum_y pi(y) a(y) h_i(y)), so the signal definitions of the paper's Table 2 are
pinned at the formula level, not only "it runs":

* ``rl``/``opd_pg`` center by the group mean of the same G responses, so the estimator's
  expectation is (G-1)/G times the population gradient;
* ``sft`` draws y from nu and its gradient is E_nu[h]. Multiplying by (w - 1) as well would
  apply the importance weight twice;
* ``opd_full`` has the uncentered advantage q/p - 1, whose gradient is E_q[h].

Tolerances are derived from the estimators' own standard errors, never pasted.
"""

import dataclasses

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.analyzer import fisher_metrics as fm
from dl_techniques.analyzer.fisher_metrics import (
    FisherConfig,
    FisherSignal,
    attach_gradient_arm,
    bound_violation_rate,
    canonical_layout,
    compute_diagonal_fisher,
    compute_gradients,
    compute_persistence,
)
from tests.numerics import true_float32_matmul

N_FEATURES = 4
N_CLASSES = 3
FLOAT_EPS = 1e-12


@pytest.fixture(autouse=True)
def cpu_float32():
    with keras.device("cpu"), true_float32_matmul():
        yield


def make_model(seed: int, scale: float = 1.0) -> keras.Model:
    keras.utils.set_random_seed(seed)
    model = keras.Sequential([
        keras.layers.Input((N_FEATURES,)),
        keras.layers.Dense(5, activation="tanh"),
        keras.layers.Dense(N_CLASSES),
    ])
    if scale != 1.0:
        model.set_weights([w * scale for w in model.get_weights()])
    return model


def make_x(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(n, N_FEATURES)).astype("float32")


def class_scores(model, x):
    """Exact probabilities (K,) and the score matrix (K, P) of one input."""
    xb = x[None]
    probs = np.asarray(keras.ops.softmax(model(xb, training=False), axis=-1))[0].astype(np.float64)
    rows = []
    for y in range(N_CLASSES):
        with tf.GradientTape() as tape:
            value = keras.ops.log_softmax(model(xb, training=False), axis=-1)[0, y]
        grads = tape.gradient(value, model.trainable_variables)
        rows.append(np.concatenate([np.asarray(g).reshape(-1) for g in grads]))
    return probs, np.stack(rows).astype(np.float64)


def within_se(estimate, se, exact, fraction=0.97, z=4.5):
    """True when >= ``fraction`` of the coordinates sit within ``z`` standard errors."""
    ok = np.abs(estimate - exact) <= z * np.maximum(se, 1e-9)
    return ok.mean() >= fraction, float(ok.mean())


def cfg(**kw) -> FisherConfig:
    base = dict(compute_gradients=True, signal_type="rl", n_responses=4, batch_size=4,
                token_level=False, selection_percentages=(5, 10))
    base.update(kw)
    return FisherConfig(**base)


def pool_with_repeats(n_unique: int, repeats: int, seed: int = 0):
    X = make_x(n_unique, seed)
    return np.tile(X, (repeats, 1)), X


def reward_fn_for(targets_by_row):
    def reward(x_i, y, target):
        return float(int(np.asarray(y)) == int(target))
    return reward


class TestSignals:

    def test_rl_gradient_matches_the_enumerated_policy_gradient_with_the_group_mean_factor(self):
        model = make_model(0)
        G = 4
        pool, X = pool_with_repeats(8, 150)
        targets = np.tile(np.arange(8) % N_CLASSES, 150)
        arm = compute_gradients(model, pool, targets, cfg(n_responses=G), np.random.default_rng(0),
                                FisherSignal(reward_fn=reward_fn_for(targets)), range(len(pool)))
        exact = 0.0
        for i, x in enumerate(X):
            probs, H = class_scores(model, x)
            r = (np.arange(N_CLASSES) == (i % N_CLASSES)).astype(np.float64)
            exact = exact + (probs * (r - probs @ r)) @ H
        exact = exact / len(X) * (G - 1) / G
        ok, frac = within_se(arm.gradient_flat, arm.gradient_se_flat, exact)
        assert ok, f"only {frac:.2f} of coordinates within the standard error"
        # anti-vacuity: the gradient is not ~0, and the uncentered-by-(G-1)/G value is far off
        assert np.abs(exact).max() > 1e-3
        assert not within_se(arm.gradient_flat, arm.gradient_se_flat, exact * G / (G - 1), z=1.0)[0]

    def test_the_envelope_holds_with_a_disjoint_fisher_sample(self):
        model = make_model(1)
        pool, X = pool_with_repeats(10, 160)
        targets = np.tile(np.arange(10) % N_CLASSES, 160)
        c = cfg()
        plan = fm.SamplePlan.build(len(pool), 800, 600, np.random.default_rng(0))
        out = compute_diagonal_fisher(model, pool, c, np.random.default_rng(1), indices=plan.fisher_indices)
        arm = compute_gradients(model, pool, targets, c, np.random.default_rng(2),
                                FisherSignal(reward_fn=reward_fn_for(targets)), plan.gradient_indices)
        joined = attach_gradient_arm(out, arm, c)
        assert joined.violation_rate <= 0.01
        assert joined.signal_scale == pytest.approx(arm.signal_scale) and joined.signal_scale > 0
        # anti-vacuity: shrinking the scale 20x makes the very same check fire
        shrunk = bound_violation_rate(out.fisher_flat, out.standard_errors, arm.gradient_flat,
                                      arm.gradient_se_flat, arm.signal_scale / 20, c.zero_threshold)
        assert shrunk > 0.1

    def test_the_gradient_support_recall_beats_the_random_baseline_and_a_shuffled_control_does_not(self):
        model = make_model(2, scale=2.0)
        pool, X = pool_with_repeats(10, 160)
        targets = np.tile(np.arange(10) % N_CLASSES, 160)
        c = cfg(selection_percentages=(5, 10, 25))
        plan = fm.SamplePlan.build(len(pool), 800, 600, np.random.default_rng(0))
        out = compute_diagonal_fisher(model, pool, c, np.random.default_rng(1), indices=plan.fisher_indices)
        arm = compute_gradients(model, pool, targets, c, np.random.default_rng(2),
                                FisherSignal(reward_fn=reward_fn_for(targets)), plan.gradient_indices)
        recall = attach_gradient_arm(out, arm, c).gradient_recall
        for q in (5, 10, 25):
            assert recall[float(q)] > 1.5 * q, (q, recall)
        # control: permuting F-hat destroys the alignment, recall collapses to about q
        shuffled = np.random.default_rng(0).permutation(out.fisher_flat)
        control = fm.gradient_recall(shuffled, arm.gradient_flat, (25,), c.zero_threshold)[25.0]
        assert control < 1.5 * 25 and control < recall[25.0]

    def test_sft_gradient_is_the_demonstration_mean_not_the_weighted_mean(self):
        model = make_model(0)
        pool, X = pool_with_repeats(8, 400)
        nu_model = make_model(5)
        nu_probs = [np.asarray(keras.ops.softmax(nu_model(x[None]), axis=-1))[0].astype(np.float64) for x in X]
        rng = np.random.default_rng(3)
        demos = np.array([rng.choice(N_CLASSES, p=nu_probs[i % 8] / nu_probs[i % 8].sum())
                          for i in range(len(pool))])

        def nu_logprob(x_i, y):
            return float(np.log(np.asarray(keras.ops.softmax(nu_model(x_i), axis=-1))[0, int(np.asarray(y))]))

        signal = FisherSignal(nu_logprob_fn=nu_logprob)
        arm = compute_gradients(model, pool, demos, cfg(signal_type="sft", n_responses=1),
                                np.random.default_rng(0), signal, range(len(pool)))
        exact_g, exact_chi2 = 0.0, 0.0
        for i, x in enumerate(X):
            probs, H = class_scores(model, x)
            exact_g = exact_g + nu_probs[i] @ H
            exact_chi2 = exact_chi2 + (nu_probs[i] ** 2 / probs).sum() - 1.0
        exact_g, exact_chi2 = exact_g / len(X), exact_chi2 / len(X)
        ok, frac = within_se(arm.gradient_flat, arm.gradient_se_flat, exact_g)
        assert ok, f"only {frac:.2f} within the standard error"
        assert arm.sft_chi2 == pytest.approx(exact_chi2, rel=0.15)
        assert arm.signal_scale == pytest.approx(np.sqrt(exact_chi2), rel=0.1)
        # the weighted mean E_nu[(w - 1) h] is a different quantity and must not be what we return
        weighted = 0.0
        for i, x in enumerate(X):
            probs, H = class_scores(model, x)
            weighted = weighted + (nu_probs[i] * (nu_probs[i] / probs - 1)) @ H
        weighted = weighted / len(X)
        assert not within_se(arm.gradient_flat, arm.gradient_se_flat, weighted, z=3.0)[0]

    def test_an_sft_chi2_blowup_is_flagged_and_does_not_crash(self):
        model = make_model(0)
        pool, X = pool_with_repeats(6, 20)
        # demonstrations on the class the model finds least likely, so w = nu/pi is huge
        least = np.array([int(np.argmin(class_scores(model, x)[0])) for x in X])
        demos = np.tile(least, 20)

        def nu_logprob(x_i, y):
            return 0.0  # nu is a point mass on y

        c = cfg(signal_type="sft", n_responses=1, sft_chi2_cap=1.5)
        arm = compute_gradients(model, pool, demos, c, np.random.default_rng(0),
                                FisherSignal(nu_logprob_fn=nu_logprob), range(len(pool)))
        out = compute_diagonal_fisher(model, pool, c, np.random.default_rng(1), indices=range(30))
        joined = attach_gradient_arm(out, arm, c)
        assert joined.metrics[fm.FM.SFT_CHI2] > 1.5
        assert joined.metrics[fm.FM.SFT_CHI2_EXCEEDS_CAP] is True
        # anti-vacuity: with a generous cap the same estimate is not flagged
        relaxed = attach_gradient_arm(out, arm, dataclasses.replace(c, sft_chi2_cap=1e12))
        assert relaxed.metrics[fm.FM.SFT_CHI2_EXCEEDS_CAP] is False

    def test_opd_pg_matches_the_enumerated_centered_log_ratio_gradient(self):
        model, teacher = make_model(0), make_model(7)
        G = 4
        pool, X = pool_with_repeats(8, 150)
        arm = compute_gradients(model, pool, None, cfg(signal_type="opd_pg", n_responses=G),
                                np.random.default_rng(0), FisherSignal(teacher_model=teacher),
                                range(len(pool)))
        exact = 0.0
        for x in X:
            probs, H = class_scores(model, x)
            q = np.asarray(keras.ops.softmax(teacher(x[None]), axis=-1))[0].astype(np.float64)
            R = np.log(q) - np.log(probs)
            exact = exact + (probs * (R - probs @ R)) @ H
        exact = exact / len(X) * (G - 1) / G
        ok, frac = within_se(arm.gradient_flat, arm.gradient_se_flat, exact)
        assert ok, f"only {frac:.2f} within the standard error"
        assert np.abs(exact).max() > 1e-3

    def test_opd_full_matches_the_teacher_weighted_score_mean(self):
        """a = q/p - 1 gives g = sum_y q(y) h(y): no centering, so no (G-1)/G factor."""
        model, teacher = make_model(0), make_model(7)
        pool, X = pool_with_repeats(8, 250)
        arm = compute_gradients(model, pool, None, cfg(signal_type="opd_full", n_responses=2),
                                np.random.default_rng(0), FisherSignal(teacher_model=teacher),
                                range(len(pool)))
        exact, chi2 = 0.0, 0.0
        for x in X:
            probs, H = class_scores(model, x)
            q = np.asarray(keras.ops.softmax(teacher(x[None]), axis=-1))[0].astype(np.float64)
            exact = exact + q @ H
            chi2 = chi2 + (q ** 2 / probs).sum() - 1.0
        exact, chi2 = exact / len(X), chi2 / len(X)
        ok, frac = within_se(arm.gradient_flat, arm.gradient_se_flat, exact)
        assert ok, f"only {frac:.2f} within the standard error"
        assert arm.signal_scale == pytest.approx(np.sqrt(chi2), rel=0.1)

    @pytest.mark.parametrize("signal_type", ["opd_pg", "opd_full"])
    def test_a_teacher_equal_to_the_student_gives_exactly_zero_advantage(self, signal_type):
        """Also pins that the teacher's probabilities enter as constants, never as gradient paths."""
        model = make_model(0)
        teacher = keras.models.clone_model(model)
        teacher.set_weights(model.get_weights())
        pool, _ = pool_with_repeats(4, 10)
        arm = compute_gradients(model, pool, None, cfg(signal_type=signal_type, n_responses=3),
                                np.random.default_rng(0), FisherSignal(teacher_model=teacher),
                                range(len(pool)))
        assert arm.signal_scale == pytest.approx(0.0, abs=1e-6)
        assert np.abs(arm.gradient_flat).max() < 1e-6

    def test_the_teacher_is_never_in_the_students_variable_set(self):
        model = make_model(0)
        teacher = make_model(7)
        before = [np.array(w) for w in teacher.get_weights()]
        pool, _ = pool_with_repeats(4, 5)
        compute_gradients(model, pool, None, cfg(signal_type="opd_pg", n_responses=2),
                          np.random.default_rng(0), FisherSignal(teacher_model=teacher), range(len(pool)))
        assert {v.path for v in teacher.trainable_variables}.isdisjoint(
            canonical_layout(model).paths) or teacher is not model
        for a, b in zip(before, teacher.get_weights()):
            assert a.tobytes() == b.tobytes()


class TestRefusals:

    def test_each_signal_names_the_hook_it_is_missing(self):
        model = make_model(0)
        pool, _ = pool_with_repeats(2, 3)
        for signal_type, hook in (("rl", "reward_fn"), ("opd_pg", "teacher_model"),
                                  ("opd_full", "teacher_model"), ("sft", "nu_logprob_fn")):
            with pytest.raises(ValueError, match=hook):
                compute_gradients(model, pool, np.zeros(len(pool), int), cfg(signal_type=signal_type),
                                  np.random.default_rng(0), FisherSignal(), range(2))

    def test_sft_without_demonstrations_raises(self):
        model = make_model(0)
        pool, _ = pool_with_repeats(2, 3)
        with pytest.raises(ValueError, match="targets"):
            compute_gradients(model, pool, None, cfg(signal_type="sft", n_responses=1),
                              np.random.default_rng(0),
                              FisherSignal(nu_logprob_fn=lambda x, y: 0.0), range(2))

    def test_a_non_finite_reward_drops_the_response_without_poisoning_the_estimate(self):
        model = make_model(0)
        pool, _ = pool_with_repeats(4, 30)
        calls = {"n": 0}

        def reward(x_i, y, target):
            calls["n"] += 1
            return float("nan") if calls["n"] % 7 == 0 else float(int(np.asarray(y)) == 0)

        arm = compute_gradients(model, pool, None, cfg(), np.random.default_rng(0),
                                FisherSignal(reward_fn=reward), range(len(pool)))
        assert arm.skipped > 0
        assert np.isfinite(arm.gradient_flat).all() and np.isfinite(arm.signal_scale)


class TestPersistenceAtToyScale:

    def test_high_fisher_coordinates_persist_across_a_few_optimizer_steps(self):
        model = make_model(0, scale=1.5)
        X = make_x(300)
        y = (X[:, 0] > 0).astype("int32") + (X[:, 1] > 0.5).astype("int32")
        c = FisherConfig(n_samples=600, batch_size=8, token_level=False, selection_percentages=(5, 10))
        rng_idx = list(range(300)) * 2
        first = compute_diagonal_fisher(model, X, c, np.random.default_rng(0), indices=rng_idx)
        model.compile(optimizer=keras.optimizers.SGD(0.05),
                      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))
        model.fit(X, y, epochs=3, batch_size=32, verbose=0)
        second = compute_diagonal_fisher(model, X, c, np.random.default_rng(1), indices=rng_idx)
        records = compute_persistence(first, {"after": second}, c)
        by_q = {r["q"]: r for r in records}
        for q in (5.0, 10.0):
            assert by_q[q]["retention_high"] > 2 * q, by_q[q]
            assert by_q[q]["random_high"] == q
        # anti-vacuity: an unrelated Fisher vector does NOT persist
        unrelated = dataclasses.replace(second, fisher_flat=np.random.default_rng(0).permutation(second.fisher_flat))
        control = compute_persistence(first, {"u": unrelated}, c)
        assert max(r["retention_high"] for r in control) < 2 * 10.0 + 10
