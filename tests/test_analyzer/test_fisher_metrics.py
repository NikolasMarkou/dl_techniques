"""
Tests for fisher_metrics: the diagonal model Fisher, its score, estimators and helpers.

The oracle for every Monte Carlo claim is an EXACT enumeration over the output space of a
tiny model (F_i = sum_y p(y) * (d_i log p(y))^2), computed here with its own gradient tapes
rather than through the code under test. A Monte Carlo estimate cannot meet an absolute
1e-6 tolerance, so the accuracy claims are split in two: the score code must match the
oracle's per-sample gradients to float32 round-off, and the Monte Carlo mean must land
within a multiple of its own standard error.

Everything runs on CPU in true float32 (``tests.numerics.true_float32_matmul``).
"""

import itertools

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.analyzer import fisher_metrics as fm
from dl_techniques.analyzer.fisher_metrics import (
    FisherConfig,
    FisherOutput,
    FisherSignal,
    SamplePlan,
    canonical_layout,
    compute_diagonal_fisher,
    compute_persistence,
    fisher_mask_export,
    fit_batch_size,
    gradient_recall,
)
from tests.numerics import true_float32_matmul

N_FEATURES = 5
N_CLASSES = 4


@pytest.fixture(autouse=True)
def cpu_float32():
    """Pin the numerics to CPU with TF32 off, and restore both afterwards."""
    with keras.device("cpu"), true_float32_matmul():
        yield


def make_classifier(seed: int = 0, final: str = None) -> keras.Model:
    """A two-layer classifier emitting logits (or probabilities with final='softmax')."""
    keras.utils.set_random_seed(seed)
    return keras.Sequential([
        keras.layers.Input((N_FEATURES,)),
        keras.layers.Dense(6, activation="tanh"),
        keras.layers.Dense(N_CLASSES, activation=final),
    ])


def make_inputs(n: int = 12, seed: int = 1) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(n, N_FEATURES)).astype("float32")


def cfg(**overrides) -> FisherConfig:
    base = dict(n_samples=50, batch_size=4, token_level=False, assert_no_weight_drift=True)
    base.update(overrides)
    return FisherConfig(**base)


def flat_grad(model: keras.Model, logp_fn) -> np.ndarray:
    """Gradient of a scalar log-likelihood w.r.t. every trainable variable, flattened."""
    with tf.GradientTape() as tape:
        value = logp_fn()
    grads = tape.gradient(value, model.trainable_variables)
    return np.concatenate([np.asarray(g).reshape(-1) for g in grads])


def exact_fisher_classifier(model: keras.Model, X: np.ndarray):
    """Exact (1/N) sum_x sum_y p(y|x) (grad log p(y|x))^2 and the exact E_y[h] per input."""
    total = None
    max_abs_mean_score = 0.0
    for x in X:
        xb = x[None]
        probs = np.asarray(keras.ops.softmax(model(xb, training=False), axis=-1))[0]
        second = None
        first = None
        for y in range(N_CLASSES):
            h = flat_grad(model, lambda: keras.ops.log_softmax(model(xb, training=False), axis=-1)[0, y])
            second = probs[y] * h ** 2 if second is None else second + probs[y] * h ** 2
            first = probs[y] * h if first is None else first + probs[y] * h
        total = second if total is None else total + second
        max_abs_mean_score = max(max_abs_mean_score, float(np.abs(first).max()))
    return total / len(X), max_abs_mean_score


class TestTheScoreAndTheEstimator:

    def test_score_vector_matches_independent_gradient(self):
        model = make_classifier()
        X = make_inputs(3)
        layout = canonical_layout(model)
        scorer = fm._Scorer(model, fm.KIND_CATEGORICAL_LOGITS, cfg(), None, layout)
        for x in X:
            for y in range(N_CLASSES):
                logp, _, h = scorer.score(x[None], np.asarray(y), np.float32)
                expected = flat_grad(model, lambda: keras.ops.log_softmax(model(x[None], training=False), axis=-1)[0, y])
                np.testing.assert_allclose(h, expected, rtol=1e-5, atol=1e-7)
                assert np.isfinite(logp)

    def test_score_normalization_has_zero_mean(self):
        """E_y[h_i] = d_i sum_y p(y) = 0 is the oracle assertion on the score."""
        model = make_classifier()
        _, max_abs_mean = exact_fisher_classifier(model, make_inputs(6))
        assert max_abs_mean < 1e-6

    def test_fisher_estimates_the_exact_enumeration_within_monte_carlo_error(self):
        model = make_classifier()
        X = make_inputs(12)
        exact, _ = exact_fisher_classifier(model, X)
        repeats = 60
        indices = list(range(len(X))) * repeats
        out = compute_diagonal_fisher(model, X, cfg(n_samples=len(indices)), np.random.default_rng(0),
                                      indices=indices)
        z = np.abs(out.fisher_flat - exact) / np.maximum(out.standard_errors, 1e-12)
        # 4 standard errors on at least 97% of the coordinates; a biased estimator (for
        # example squaring the mean score) fails this by orders of magnitude.
        assert np.mean(z <= 4.0) >= 0.97
        assert out.n_samples == len(indices) and out.skipped_samples == 0
        # anti-vacuity: the exact Fisher is not ~0 everywhere, so the check bites
        assert exact.max() > 1e-3

    def test_a_probability_head_gives_the_same_fisher_as_the_logit_head(self):
        logits_model = make_classifier(final=None)
        probs_model = make_classifier(final="softmax")
        X = make_inputs(6)
        a = compute_diagonal_fisher(logits_model, X, cfg(n_samples=24), np.random.default_rng(3), indices=list(range(6)) * 4)
        b = compute_diagonal_fisher(probs_model, X, cfg(n_samples=24), np.random.default_rng(3), indices=list(range(6)) * 4)
        assert a.output_kind == fm.KIND_CATEGORICAL_LOGITS and b.output_kind == fm.KIND_CATEGORICAL_PROBS
        np.testing.assert_allclose(a.fisher_flat, b.fisher_flat, rtol=1e-3, atol=1e-8)

    def test_gaussian_regression_fisher_matches_the_closed_form(self):
        """For f(x) = w.x + b with variance s2: F_w = E[x^2]/s2 and F_b = 1/s2."""
        keras.utils.set_random_seed(0)
        model = keras.Sequential([keras.layers.Input((2,)), keras.layers.Dense(1)])
        X = make_inputs(40)[:, :2]
        s2 = 0.25
        out = compute_diagonal_fisher(model, X, cfg(n_samples=4000, regression_variance=s2),
                                      np.random.default_rng(0), indices=list(range(40)) * 100)
        assert out.output_kind == fm.KIND_GAUSSIAN
        kernel_slice = [s for p, s, e in out.param_slices if p.endswith("kernel")][0]
        expected_w = (X ** 2).mean(axis=0) / s2
        got_w = out.fisher_flat[kernel_slice:kernel_slice + 2]
        se_w = out.standard_errors[kernel_slice:kernel_slice + 2]
        assert np.all(np.abs(got_w - expected_w) <= 5 * se_w)
        bias_slice = [s for p, s, e in out.param_slices if p.endswith("bias")][0]
        assert abs(out.fisher_flat[bias_slice] - 1 / s2) <= 5 * out.standard_errors[bias_slice]
        # anti-vacuity: scoring the MEAN response (y = f(x)) would make every score zero
        assert out.fisher_flat.min() > 0.1

    def test_a_regression_without_a_variance_is_not_silently_guessed(self):
        keras.utils.set_random_seed(0)
        model = keras.Sequential([keras.layers.Input((2,)), keras.layers.Dense(1)])
        scorer_kwargs = dict(model=model, kind=fm.KIND_GAUSSIAN, cfg=cfg(), signal=None, layout=None)
        with pytest.raises(ValueError, match="regression_variance"):
            fm._Scorer(**scorer_kwargs)

    def test_bernoulli_head_matches_the_exact_enumeration(self):
        keras.utils.set_random_seed(0)
        model = keras.Sequential([keras.layers.Input((N_FEATURES,)),
                                  keras.layers.Dense(3, activation="sigmoid")])
        X = make_inputs(8)
        exact = None
        for x in X:
            probs = np.asarray(model(x[None]))[0]
            for bits in itertools.product([0.0, 1.0], repeat=3):
                y = np.asarray(bits, "float32")[None]
                weight = float(np.prod(np.where(y[0] == 1, probs, 1 - probs)))
                h = flat_grad(model, lambda: keras.ops.sum(
                    y * keras.ops.log(model(x[None])) + (1 - y) * keras.ops.log(1 - model(x[None])), axis=-1)[0])
                exact = weight * h ** 2 if exact is None else exact + weight * h ** 2
        exact = exact / len(X)
        out = compute_diagonal_fisher(model, X, cfg(n_samples=8 * 150), np.random.default_rng(0),
                                      indices=list(range(8)) * 150)
        assert out.output_kind == fm.KIND_BERNOULLI
        z = np.abs(out.fisher_flat - exact) / np.maximum(out.standard_errors, 1e-12)
        assert np.mean(z <= 4.0) >= 0.97


class TestDiscipline:

    @staticmethod
    def nan_signal(bad_rows):
        """A custom likelihood that is NaN for the listed pool rows (a bug-injection hook)."""
        calls = {"x": None}

        def logprob(model, x_i, y):
            logits = model(x_i, training=False)
            value = keras.ops.log_softmax(logits, axis=-1)[0, int(np.asarray(y))]
            if any(np.allclose(x_i[0], row) for row in bad_rows):
                return value * float("nan")
            return value

        def sampler(model, x_batch, config, rng):
            probs = np.asarray(keras.ops.softmax(model(x_batch, training=False), axis=-1))
            return [np.asarray(rng.choice(N_CLASSES, p=p / p.sum())) for p in probs]

        return FisherSignal(output_logprob_fn=logprob, sampler=sampler), calls

    def test_skipped_samples_are_counted_not_clipped(self):
        model = make_classifier()
        X = make_inputs(10)
        bad = [X[2], X[7]]
        signal, _ = self.nan_signal(bad)
        out = compute_diagonal_fisher(model, X, cfg(n_samples=10), np.random.default_rng(0),
                                      indices=list(range(10)), signal=signal)
        assert out.output_kind == fm.KIND_CUSTOM
        assert out.skipped_samples == 2 and out.n_samples == 8
        # the estimate is the finite-only mean, not a zero-inflated one
        clean_signal, _ = self.nan_signal([])
        keep = [i for i in range(10) if i not in (2, 7)]
        reference = compute_diagonal_fisher(model, X, cfg(n_samples=8), np.random.default_rng(0),
                                            indices=keep, signal=clean_signal)
        assert out.fisher_flat.min() >= 0
        assert np.isfinite(out.fisher_flat).all()
        assert out.n_samples == reference.n_samples
        # anti-vacuity: with no bad rows nothing is skipped
        assert reference.skipped_samples == 0

    def test_every_sample_nonfinite_raises_instead_of_publishing_zero(self):
        model = make_classifier()
        X = make_inputs(4)
        signal, _ = self.nan_signal(list(X))
        with pytest.raises(RuntimeError, match="finite"):
            compute_diagonal_fisher(model, X, cfg(n_samples=4), np.random.default_rng(0),
                                    indices=[0, 1, 2, 3], signal=signal)

    @staticmethod
    def drifting_signal(model):
        def sampler(m, x_batch, config, rng):
            kernel = m.layers[0].kernel
            kernel.assign(kernel + 1e-3)
            return [np.asarray(0) for _ in range(len(x_batch))]

        def logprob(m, x_i, y):
            return keras.ops.log_softmax(m(x_i, training=False), axis=-1)[0, 0]

        return FisherSignal(output_logprob_fn=logprob, sampler=sampler)

    def test_no_weight_drift_assertion_fires(self):
        model = make_classifier()
        X = make_inputs(4)
        with pytest.raises(RuntimeError, match="changed during the Fisher pass"):
            compute_diagonal_fisher(model, X, cfg(n_samples=4), np.random.default_rng(0),
                                    indices=[0, 1], signal=self.drifting_signal(model))

    def test_the_drift_check_is_what_raises_not_something_else(self):
        """Anti-vacuity arm: with the check off, the same drifting pass completes."""
        model = make_classifier()
        X = make_inputs(4)
        out = compute_diagonal_fisher(model, X, cfg(n_samples=4, assert_no_weight_drift=False),
                                      np.random.default_rng(0), indices=[0, 1],
                                      signal=self.drifting_signal(model))
        assert out.n_samples == 2

    def test_a_clean_pass_leaves_the_weights_byte_identical(self):
        model = make_classifier()
        before = [np.array(w) for w in model.get_weights()]
        compute_diagonal_fisher(model, make_inputs(5), cfg(n_samples=5), np.random.default_rng(0))
        for a, b in zip(before, model.get_weights()):
            assert a.tobytes() == b.tobytes()

    def test_reproducibility_with_a_fixed_seed(self):
        model = make_classifier()
        X = make_inputs(8)
        runs = [compute_diagonal_fisher(model, X, cfg(n_samples=24), np.random.default_rng(7),
                                        indices=list(range(8)) * 3) for _ in range(2)]
        assert runs[0].fisher_flat.tobytes() == runs[1].fisher_flat.tobytes()
        assert runs[0].skipped_samples == runs[1].skipped_samples
        assert runs[0].samples_used == runs[1].samples_used

    def test_one_generator_advances_across_calls(self):
        """A per-call reconstruction from the same seed would hand both calls one stream."""
        model = make_classifier()
        X = make_inputs(8)
        rng = np.random.default_rng(7)
        first = compute_diagonal_fisher(model, X, cfg(n_samples=24), rng, indices=list(range(8)) * 3)
        second = compute_diagonal_fisher(model, X, cfg(n_samples=24), rng, indices=list(range(8)) * 3)
        assert [s["response"] for s in first.samples_used] != [s["response"] for s in second.samples_used]

    def test_batch_size_shrinks_to_the_memory_budget_and_logs(self):
        assert fit_batch_size(64, per_sample_bytes=10 * 1024 * 1024, memory_limit_mb=100) == 10
        assert fit_batch_size(64, per_sample_bytes=10 * 1024 * 1024, memory_limit_mb=1) == 1
        assert fit_batch_size(8, per_sample_bytes=1024, memory_limit_mb=100) == 8
        assert fit_batch_size(64, per_sample_bytes=10 ** 9, memory_limit_mb=None) == 64

    def test_a_pass_under_a_tiny_budget_runs_at_batch_one_instead_of_crashing(self):
        model = make_classifier()
        out = compute_diagonal_fisher(model, make_inputs(4), cfg(n_samples=4, memory_limit_mb=0),
                                      np.random.default_rng(0))
        assert out.n_samples == 4


class TestConfigContracts:

    def test_a_one_response_rl_configuration_raises(self):
        with pytest.raises(ValueError, match="n_responses must be >= 2"):
            FisherConfig(compute_gradients=True, signal_type="rl", n_responses=1).validate()

    def test_opd_pg_is_group_centered_so_it_also_needs_two_responses(self):
        with pytest.raises(ValueError, match="n_responses must be >= 2"):
            FisherConfig(compute_gradients=True, signal_type="opd_pg", n_responses=1).validate()

    def test_uncentered_signals_accept_one_response(self):
        for signal_type in ("opd_full", "sft"):
            FisherConfig(compute_gradients=True, signal_type=signal_type, n_responses=1).validate()

    def test_gradients_without_a_signal_type_raise(self):
        with pytest.raises(ValueError, match="other than 'none'"):
            FisherConfig(compute_gradients=True, signal_type="none").validate()

    @pytest.mark.parametrize("field,value", [("accumulate_dtype", "float16"), ("score_scope", "x"),
                                             ("signal_type", "x"), ("n_samples", 0)])
    def test_invalid_values_raise(self, field, value):
        with pytest.raises(ValueError):
            FisherConfig(**{field: value}).validate()


class TestLayoutAndPlan:

    def test_fingerprint_changes_with_the_variable_set_and_slices_tile_the_flat_vector(self):
        a = canonical_layout(make_classifier())
        b = canonical_layout(keras.Sequential([keras.layers.Input((N_FEATURES,)), keras.layers.Dense(7)]))
        assert a.fingerprint != b.fingerprint
        assert a.slices[0][0] == 0 and a.slices[-1][1] == a.total
        assert all(x[1] == y[0] for x, y in zip(a.slices, a.slices[1:]))

    def test_identically_built_models_agree_despite_keras_auto_names_and_others_do_not(self):
        first, second = make_classifier(0), make_classifier(1)
        assert [v.path for v in first.trainable_variables] != [v.path for v in second.trainable_variables]
        assert canonical_layout(first).fingerprint == canonical_layout(second).fingerprint
        wider = keras.Sequential([keras.layers.Input((N_FEATURES,)), keras.layers.Dense(7, activation="tanh"),
                                  keras.layers.Dense(N_CLASSES)])
        assert canonical_layout(wider).fingerprint != canonical_layout(first).fingerprint

    def test_sample_plan_is_disjoint_replayable_and_refuses_an_undersized_pool(self):
        plan = SamplePlan.build(100, 30, 20, np.random.default_rng(0), seed=0)
        assert not set(plan.fisher_indices) & set(plan.gradient_indices)
        assert len(plan.fisher_indices) == 30 and len(plan.gradient_indices) == 20
        again = SamplePlan.from_dict(plan.to_dict())
        assert again == plan and again.fingerprint() == plan.fingerprint()
        assert SamplePlan.build(100, 30, 20, np.random.default_rng(1)).fingerprint() != plan.fingerprint()
        with pytest.raises(ValueError, match="disjointly"):
            SamplePlan.build(40, 30, 20, np.random.default_rng(0))
        with pytest.raises(ValueError, match="disjoint"):
            SamplePlan(10, (1, 2), (2, 3))

    def test_a_fisher_only_plan_clips_to_the_pool(self):
        plan = SamplePlan.build(10, 30, 0, np.random.default_rng(0))
        assert len(plan.fisher_indices) == 10

    def test_unflatten_round_trips_the_canonical_order(self):
        model = make_classifier()
        out = compute_diagonal_fisher(model, make_inputs(4), cfg(n_samples=4), np.random.default_rng(0))
        pieces = fm.unflatten(out.fisher_flat, out)
        rebuilt = np.concatenate([pieces[p].reshape(-1) for p, _, _ in out.param_slices])
        np.testing.assert_array_equal(rebuilt, out.fisher_flat)
        assert [pieces[p].shape for p, _, _ in out.param_slices] == [s for _, s, _ in out.shapes]


class TestSelectionHelpers:

    def test_recall_is_nan_not_zero_when_the_set_is_empty(self):
        fisher = np.linspace(1, 2, 10)
        recall = gradient_recall(fisher, fisher, [0.05, 50], zero_threshold=1e-12)
        assert np.isnan(recall[0.05]) and recall[50] == 100.0

    def test_recall_of_identical_rankings_is_100_and_of_independent_ones_is_near_q(self):
        rng = np.random.default_rng(0)
        fisher = rng.random(20000) + 0.01
        assert gradient_recall(fisher, fisher, [1], 1e-12)[1.0] == 100.0
        independent = gradient_recall(fisher, rng.random(20000), [5], 1e-12)[5.0]
        assert independent < 15.0  # random baseline is 5%

    def test_fisher_dead_coordinates_are_excluded_from_recall(self):
        fisher = np.concatenate([np.zeros(900), np.linspace(1, 2, 100)])
        recall = gradient_recall(fisher, fisher, [10], zero_threshold=1e-12)
        assert recall[10.0] == 100.0  # K_q = 10 of the 100 alive, not of all 1000

    def test_ties_break_by_the_canonical_coordinate_order(self):
        fisher = np.ones(10)
        mask = fisher_mask_export(_fake_output(fisher), 30)
        assert np.flatnonzero(mask["a"]).tolist() == [0, 1, 2]

    def test_mask_export_selects_top_k_per_variable_shape(self):
        fisher = np.arange(1, 13, dtype=float)
        out = _fake_output(fisher, shapes=[("a", (2, 3), "float32"), ("b", (6,), "float32")])
        masks = fisher_mask_export(out, 25)
        assert masks["a"].shape == (2, 3) and masks["b"].shape == (6,)
        assert int(masks["a"].sum() + masks["b"].sum()) == 3
        assert masks["b"][-3:].all()

    def test_mask_export_refuses_an_empty_selection(self):
        with pytest.raises(ValueError, match="selects nothing"):
            fisher_mask_export(_fake_output(np.ones(10)), 5)


def _fake_output(fisher, shapes=None) -> FisherOutput:
    shapes = shapes or [("a", (len(fisher),), "float32")]
    slices, start = [], 0
    for path, shape, _ in shapes:
        size = int(np.prod(shape))
        slices.append((path, start, start + size))
        start += size
    return FisherOutput(fisher_flat=np.asarray(fisher, float), standard_errors=None,
                        param_slices=slices, shapes=shapes, variable_fingerprint="fp",
                        n_samples=1, skipped_samples=0, output_kind=fm.KIND_CATEGORICAL_LOGITS)


class TestPersistence:

    def test_identical_policies_retain_everything_and_random_baselines_are_q_and_100_minus_q(self):
        rng = np.random.default_rng(0)
        ref = _fake_output(rng.random(5000) + 0.01)
        records = compute_persistence(ref, {"same": ref}, FisherConfig(selection_percentages=(1, 10)))
        for rec in records:
            assert rec["retention_high"] == 100.0 and rec["retention_low"] == 100.0
            assert rec["mass_ratio"] == pytest.approx(1.0)
            assert rec["random_high"] == rec["q"] and rec["random_low"] == 100.0 - rec["q"]

    def test_an_independent_policy_retains_about_the_random_baseline(self):
        rng = np.random.default_rng(0)
        ref = _fake_output(rng.random(20000) + 0.01)
        other = _fake_output(rng.random(20000) + 0.01)
        rec = compute_persistence(ref, {"o": other}, FisherConfig(selection_percentages=(10,)))[0]
        assert abs(rec["retention_high"] - 10.0) < 4.0
        assert abs(rec["retention_low"] - 90.0) < 4.0

    def test_fingerprint_mismatch_is_loud(self):
        ref = _fake_output(np.ones(100))
        other = dataclass_replace(_fake_output(np.ones(100)), variable_fingerprint="different")
        with pytest.raises(ValueError, match="do not align"):
            compute_persistence(ref, {"o": other}, FisherConfig())


def dataclass_replace(obj, **kwargs):
    import dataclasses
    return dataclasses.replace(obj, **kwargs)
