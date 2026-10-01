"""Tests for :mod:`dl_techniques.analyzer.representation_metrics`.

Oracles are closed-form constructions (planted variance, exact parallelograms,
hand-built accuracy curves) and an independent per-point silhouette, so none
shares code with the implementation.
"""

import numpy as np
import pytest

from dl_techniques.analyzer import representation_metrics as rm


def _planted(variances, n=400, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, len(variances))) * np.sqrt(variances)
    q, _ = np.linalg.qr(rng.normal(size=(len(variances), len(variances))))
    return z @ q.T


def test_explained_variance_recovers_planted_spectrum():
    x = _planted([9.0, 4.0, 1.0, 1.0, 1.0], n=20000)
    r = rm.explained_variance_ratios(x)
    np.testing.assert_allclose(r, np.array([9, 4, 1, 1, 1]) / 16, atol=0.02)
    assert r.sum() == pytest.approx(1.0)
    assert np.all(np.diff(r) <= 1e-12)


def test_cumulative_and_top_k():
    x = _planted([5.0, 3.0, 1.0, 1.0])
    cum = rm.cumulative_explained_variance(x)
    assert cum[-1] == pytest.approx(1.0)
    assert rm.top_k_explained_variance(x, 2) == pytest.approx(cum[1])
    assert rm.top_k_explained_variance(x, 99) == pytest.approx(1.0)


def test_planar_points_are_100_percent_in_two_components():
    rng = np.random.default_rng(1)
    plane = rng.normal(size=(50, 2)) @ rng.normal(size=(2, 16))
    assert rm.top_k_explained_variance(plane, 2) == pytest.approx(1.0)


def test_identical_points_have_zero_variance_not_nan():
    assert rm.explained_variance_ratios(np.ones((5, 3))).tolist() == [0.0, 0.0, 0.0]


@pytest.mark.parametrize("bad", [np.zeros(5), np.zeros((1, 3))])
def test_bad_embedding_shapes_raise(bad):
    with pytest.raises(ValueError):
        rm.explained_variance_ratios(bad)


def test_exact_parallelograms_have_zero_loss():
    # i -> j and m -> n share the offset t, in a 2-D table (PCA is lossless).
    rng = np.random.default_rng(2)
    base = rng.normal(size=(10, 2))
    t = np.array([1.5, -0.7])
    table = np.concatenate([base, base + t])  # rows k and k+10 are related
    quads = np.array([[0, 10, 3, 13], [1, 11, 7, 17]])
    np.testing.assert_allclose(rm.parallelogram_loss(table, quads), 0.0, atol=1e-9)


def test_broken_parallelogram_has_positive_loss():
    rng = np.random.default_rng(3)
    base = rng.normal(size=(10, 2))
    table = np.concatenate([base, base + np.array([1.5, -0.7])])
    table[13] += np.array([2.0, 2.0])
    loss = rm.parallelogram_loss(table, np.array([[0, 10, 3, 13], [0, 10, 4, 14]]))
    assert loss[0] > 0.3 and loss[1] < 1e-9


def test_parallelogram_loss_matches_eq3_by_hand_and_is_scale_invariant():
    rng = np.random.default_rng(4)
    table = rng.normal(size=(30, 8))
    quads = rm.sample_parallelogram_quadruples([(0, 1), (2, 3), (4, 5), (6, 7)], 50, seed=1)
    got = rm.parallelogram_loss(table, quads)
    e = rm.pca_project(table, 2)
    sigma = np.sqrt((e ** 2).sum(1).mean())
    i, j, m, n = quads.T
    expected = np.array([np.linalg.norm(e[a] + e[d] - e[b] - e[c]) / sigma
                         for a, b, c, d in zip(i, j, m, n)])
    np.testing.assert_allclose(got, expected, atol=1e-10)
    np.testing.assert_allclose(rm.parallelogram_loss(7.0 * table, quads), got, atol=1e-9)


def test_parallelogram_loss_validation():
    t = np.random.default_rng(5).normal(size=(6, 4))
    with pytest.raises(ValueError):
        rm.parallelogram_loss(t, np.zeros((3, 3), int))
    with pytest.raises(ValueError):
        rm.parallelogram_loss(t, np.array([[0, 1, 2, 6]]))
    with pytest.raises(ValueError):
        rm.parallelogram_loss(np.ones((6, 4)), np.array([[0, 1, 2, 3]]))


def test_sampled_quadruples_use_two_distinct_pairs_deterministically():
    pairs = [(0, 1), (2, 3), (4, 5)]
    q = rm.sample_parallelogram_quadruples(pairs, 200, seed=7)
    assert q.shape == (200, 4)
    assert (q[:, :2] != q[:, 2:]).any(axis=1).all()
    pair_set = {tuple(p) for p in pairs}
    assert all((a, b) in pair_set and (c, d) in pair_set for a, b, c, d in q)
    np.testing.assert_array_equal(q, rm.sample_parallelogram_quadruples(pairs, 200, seed=7))
    with pytest.raises(ValueError):
        rm.sample_parallelogram_quadruples([(0, 1)], 5)


def _brute_silhouette(x, y):
    d = np.linalg.norm(x[:, None] - x[None], axis=-1)
    s = []
    for p in range(len(x)):
        same = (y == y[p]) & (np.arange(len(x)) != p)
        a = d[p, same].mean() if same.any() else 0.0
        b = min(d[p, y == c].mean() for c in set(y) if c != y[p])
        s.append(0.0 if not same.any() else (b - a) / max(a, b))
    return float(np.mean(s))


def test_silhouette_matches_brute_force_and_ranks_the_true_partition_first():
    rng = np.random.default_rng(6)
    centers = np.array([[0, 0], [10, 0], [0, 10.0]])
    true = np.repeat([0, 1, 2], 20)
    x = centers[true] + rng.normal(scale=0.5, size=(60, 2))
    assert rm.partition_silhouette(x, true) == pytest.approx(_brute_silhouette(x, true), abs=1e-9)
    ranked = rm.rank_partitions(x, {"random": rng.integers(0, 3, 60), "true": true, "one": np.zeros(60, int)})
    assert ranked[0][0] == "true" and ranked[0][1] > 0.8
    assert ranked[-1][0] == "one" and np.isnan(ranked[-1][1])


def test_silhouette_undefined_cases_are_nan():
    x = np.random.default_rng(0).normal(size=(5, 2))
    assert np.isnan(rm.partition_silhouette(x, [0] * 5))
    assert np.isnan(rm.partition_silhouette(x, range(5)))
    with pytest.raises(ValueError):
        rm.partition_silhouette(x, [0, 1])


def test_epochs_to_threshold_needs_a_full_consecutive_run():
    acc = [0.1, 0.95, 0.95, 0.5, 0.95, 0.95, 0.95, 0.95]
    assert rm.epochs_to_threshold(acc, 0.9, consecutive=3) == 4
    assert rm.epochs_to_threshold(acc, 0.9, consecutive=2) == 1
    assert rm.epochs_to_threshold(acc, 0.9, consecutive=5) is None
    assert rm.epochs_to_threshold([0.9, 0.9], 0.9, consecutive=1) is None  # strict >
    with pytest.raises(ValueError):
        rm.epochs_to_threshold(acc, consecutive=0)


def test_grokking_gap():
    train = [0.2] + [0.99] * 30
    delayed = [0.1] * 20 + [0.99] * 11
    together = [0.2] + [0.99] * 30
    never = [0.1] * 31
    assert rm.grokking_gap(train, delayed, consecutive=5) == 19
    assert rm.grokking_gap(train, together, consecutive=5) == 0
    assert rm.grokking_gap(train, never, consecutive=5) is None
