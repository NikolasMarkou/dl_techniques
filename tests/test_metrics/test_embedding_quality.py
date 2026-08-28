"""Tests for the pool-level embedding-quality metrics.

Every ranking assertion below uses a **planted** similarity matrix with
hand-computed expected values, so the oracle is independent of the
implementation rather than derived from it.
"""

import math

import numpy as np
import pytest

from dl_techniques.metrics.embedding_quality import (
    alignment,
    anisotropy,
    effective_rank,
    embedding_norm_stats,
    l2_normalize,
    mrr_at_k,
    ndcg_at_k,
    rank_of_ground_truth,
    ranking_metrics,
    recall_at_k,
    recall_at_ks,
    uniformity,
)


def planted_similarity(target_ranks, n_candidates=10):
    """Build a similarity matrix whose gold ranks are exactly `target_ranks`.

    Row `i` gets strictly descending scores; the gold column is placed at
    position `target_ranks[i] - 1`, so the rank is chosen rather than measured.
    """
    n_queries = len(target_ranks)
    sims = np.zeros((n_queries, n_candidates), dtype=np.float64)
    truth = np.zeros(n_queries, dtype=np.int64)
    for i, rank in enumerate(target_ranks):
        scores = np.arange(n_candidates, 0, -1, dtype=np.float64)
        order = np.random.default_rng(i).permutation(n_candidates)
        sims[i, order] = scores
        truth[i] = int(order[rank - 1])
    return sims, truth


# ---------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------

class TestPlantedRankOracle:
    """Closed-form values on ranks that were chosen, not measured."""

    RANKS = [1, 2, 4, 8]

    def test_the_planted_ranks_are_recovered_exactly(self):
        sims, truth = planted_similarity(self.RANKS)
        np.testing.assert_array_equal(
            rank_of_ground_truth(sims, truth), np.array(self.RANKS)
        )

    def test_recall(self):
        ranks = np.array(self.RANKS)
        assert recall_at_k(ranks, 1) == pytest.approx(0.25)
        assert recall_at_k(ranks, 5) == pytest.approx(0.75)
        assert recall_at_k(ranks, 10) == pytest.approx(1.0)

    def test_mrr_hand_computed(self):
        ranks = np.array(self.RANKS)
        assert mrr_at_k(ranks, 10) == pytest.approx((1 + 1 / 2 + 1 / 4 + 1 / 8) / 4)
        assert mrr_at_k(ranks, 5) == pytest.approx((1 + 1 / 2 + 1 / 4 + 0) / 4)

    def test_ndcg_hand_computed(self):
        ranks = np.array(self.RANKS)
        expected = (
            1 / math.log2(2) + 1 / math.log2(3)
            + 1 / math.log2(5) + 1 / math.log2(9)
        ) / 4
        assert ndcg_at_k(ranks, 10) == pytest.approx(expected)

    def test_ranking_metrics_bundle(self):
        sims, truth = planted_similarity(self.RANKS)
        out = ranking_metrics(sims, truth)
        assert out["recall_at_1"] == pytest.approx(0.25)
        assert out["median_rank"] == pytest.approx(3.0)
        assert out["mean_rank"] == pytest.approx(15 / 4)
        assert out["n_queries"] == 4
        assert out["n_candidates"] == 10
        assert out["chance_recall_at_1"] == pytest.approx(0.1)


class TestTiesArePessimistic:
    """A fully collapsed encoder must score zero, not one.

    This is the guard that stops a degenerate model from looking perfect. Under
    optimistic or midpoint tie-breaking an all-equal similarity matrix hands
    every query rank 1 and `recall@1 == 1.0`.
    """

    def test_a_totally_tied_matrix_scores_zero_recall(self):
        sims = np.ones((5, 5))
        truth = np.arange(5)
        np.testing.assert_array_equal(
            rank_of_ground_truth(sims, truth), np.full(5, 5)
        )
        assert recall_at_k(rank_of_ground_truth(sims, truth), 1) == 0.0
        assert recall_at_k(rank_of_ground_truth(sims, truth), 4) == 0.0

    def test_a_partial_tie_counts_every_tied_candidate_against_the_gold(self):
        # Gold ties with two others, one candidate is strictly better.
        sims = np.array([[5.0, 3.0, 3.0, 3.0]])
        ranks = rank_of_ground_truth(sims, np.array([1]))
        assert ranks[0] == 4  # 1 strictly better + 2 tied


class TestChunkingIsExact:
    """A chunk boundary must not drop or double-count a candidate."""

    def test_identical_ranks_at_every_chunk_size(self):
        rng = np.random.default_rng(0)
        sims = rng.standard_normal((37, 23))
        truth = rng.integers(0, 23, 37)
        reference = rank_of_ground_truth(sims, truth, chunk_size=512)
        for chunk in (1, 2, 7, 36, 37, 1000):
            np.testing.assert_array_equal(
                rank_of_ground_truth(sims, truth, chunk_size=chunk), reference
            )


class TestRankingValidation:
    def test_rejects_a_non_2d_similarity(self):
        with pytest.raises(ValueError, match="2-D"):
            rank_of_ground_truth(np.zeros(5), np.zeros(5, dtype=int))

    def test_rejects_a_length_mismatch(self):
        with pytest.raises(ValueError, match="entries"):
            rank_of_ground_truth(np.zeros((3, 4)), np.zeros(2, dtype=int))

    def test_rejects_an_out_of_range_index(self):
        with pytest.raises(ValueError, match=r"\[0, 4\)"):
            rank_of_ground_truth(np.zeros((2, 4)), np.array([0, 9]))

    @pytest.mark.parametrize("fn", [recall_at_k, mrr_at_k, ndcg_at_k])
    def test_rejects_a_non_positive_k(self, fn):
        with pytest.raises(ValueError, match="k must be"):
            fn(np.array([1, 2]), 0)

    @pytest.mark.parametrize("fn", [recall_at_k, mrr_at_k, ndcg_at_k])
    def test_empty_input_is_nan_not_a_crash(self, fn):
        assert math.isnan(fn(np.array([]), 5))


class TestTheThreeRankingMetricsAreOneMeasurement:
    """Pins the redundancy the report relies on, and its limit."""

    def test_mrr_is_the_documented_linear_functional_of_the_recall_curve(self):
        """MRR@k = sum_j c_j * R@j with c_j = 1/j - 1/(j+1), c_k = 1/k."""
        rng = np.random.default_rng(3)
        ranks = rng.integers(1, 200, 500)
        k = 10
        curve = np.array([recall_at_k(ranks, j) for j in range(1, k + 1)])
        coeffs = np.array(
            [1 / j - 1 / (j + 1) for j in range(1, k)] + [1 / k]
        )
        assert float(coeffs @ curve) == pytest.approx(mrr_at_k(ranks, k), abs=1e-12)

    def test_the_means_of_mrr_and_ndcg_can_disagree(self):
        """Per query they are rank-equivalent; their MEANS are not.

        Measured counterexample, k=100: system A ranks [1, 100], system B
        ranks [2, 2]. MRR prefers A, nDCG prefers B.
        """
        a, b, k = np.array([1, 100]), np.array([2, 2]), 100
        assert mrr_at_k(a, k) > mrr_at_k(b, k)
        assert ndcg_at_k(a, k) < ndcg_at_k(b, k)


# ---------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------

class TestAnisotropy:
    def test_a_collapsed_space_scores_one(self):
        v = np.random.default_rng(0).standard_normal(16)
        assert anisotropy(np.tile(v, (100, 1))) == pytest.approx(1.0, abs=1e-9)

    def test_an_antipodal_pair_scores_minus_one(self):
        v = np.random.default_rng(0).standard_normal(8)
        assert anisotropy(np.stack([v, -v])) == pytest.approx(-1.0, abs=1e-9)

    def test_an_orthonormal_basis_scores_zero(self):
        assert anisotropy(np.eye(8)) == pytest.approx(0.0, abs=1e-12)

    def test_an_isotropic_gaussian_sample_is_near_zero(self):
        x = np.random.default_rng(1).standard_normal((4096, 64))
        assert abs(anisotropy(x)) < 0.02

    def test_the_fast_identity_matches_a_brute_force_double_loop(self):
        """The O(nd) closed form is where a silent bug would live."""
        x = np.random.default_rng(2).standard_normal((30, 8))
        u = l2_normalize(x)
        n = u.shape[0]
        brute = sum(
            float(u[i] @ u[j]) for i in range(n) for j in range(n) if i != j
        ) / (n * (n - 1))
        assert anisotropy(x) == pytest.approx(brute, abs=1e-10)

    def test_rejects_a_single_embedding(self):
        with pytest.raises(ValueError, match="at least 2"):
            anisotropy(np.ones((1, 4)))

    def test_rejects_a_zero_norm_row(self):
        x = np.eye(3)
        x[1] = 0.0
        with pytest.raises(ValueError, match="zero norm"):
            anisotropy(x)


class TestEffectiveRank:
    def test_an_orthonormal_basis_has_effective_rank_d(self):
        assert effective_rank(np.eye(8), center=False) == pytest.approx(8.0, abs=1e-9)

    def test_a_rank_one_matrix_has_effective_rank_one(self):
        rng = np.random.default_rng(0)
        x = np.outer(rng.standard_normal(50), rng.standard_normal(12))
        assert effective_rank(x, center=False) == pytest.approx(1.0, abs=1e-6)

    def test_it_can_never_exceed_the_true_rank(self):
        rng = np.random.default_rng(1)
        x = rng.standard_normal((100, 3)) @ rng.standard_normal((3, 16))
        value = effective_rank(x, center=False)
        assert value <= 3.0 + 1e-6
        assert value > 2.0

    def test_an_all_zero_matrix_returns_zero_not_nan(self):
        assert effective_rank(np.zeros((10, 4))) == 0.0

    def test_centering_raises_the_rank_on_offset_data(self):
        """A mean offset is one dominant direction, and it swamps the spectrum.

        MEASURED on `standard_normal((100, 8)) + 50`: the uncentered singular
        values are [1412.9, 11.6, 11.1, ...] -- the offset alone contributes a
        value two orders above the rest -- giving effective rank 1.33, while
        centering gives 7.93. So `center=True` REPORTS A HIGHER rank here, not
        a lower one, and the flag is not cosmetic: on offset data the two
        answers differ by ~6x and disagree about whether the space has
        collapsed.
        """
        rng = np.random.default_rng(2)
        x = rng.standard_normal((100, 8)) + 50.0
        centered = effective_rank(x, center=True)
        uncentered = effective_rank(x, center=False)
        assert uncentered == pytest.approx(1.33, abs=0.05)
        assert centered == pytest.approx(7.93, abs=0.05)
        assert centered > uncentered


class TestAlignmentAndUniformity:
    def test_alignment_of_identical_pairs_is_zero(self):
        x = np.random.default_rng(0).standard_normal((32, 16))
        assert alignment(x, x) == pytest.approx(0.0, abs=1e-12)

    def test_alignment_of_antipodal_pairs_is_four(self):
        x = np.random.default_rng(0).standard_normal((32, 16))
        assert alignment(x, -x) == pytest.approx(4.0, abs=1e-9)

    def test_alignment_is_affine_in_the_mean_positive_cosine(self):
        rng = np.random.default_rng(1)
        a, b = rng.standard_normal((64, 16)), rng.standard_normal((64, 16))
        mean_cos = float(np.mean(np.sum(l2_normalize(a) * l2_normalize(b), axis=1)))
        assert alignment(a, b) == pytest.approx(2 - 2 * mean_cos, abs=1e-9)

    def test_alignment_rejects_a_shape_mismatch(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            alignment(np.zeros((4, 8)), np.zeros((5, 8)))

    def test_a_collapsed_space_has_the_worst_possible_uniformity(self):
        v = np.random.default_rng(0).standard_normal(16)
        assert uniformity(np.tile(v, (64, 1))) == pytest.approx(0.0, abs=1e-9)

    def test_a_spread_space_scores_better_than_a_clustered_one(self):
        """The direction the report depends on."""
        rng = np.random.default_rng(2)
        spread = rng.standard_normal((256, 32))
        clustered = rng.standard_normal((256, 32)) * 0.01 + 10.0
        assert uniformity(spread) < uniformity(clustered)

    def test_subsampling_without_an_rng_is_refused(self):
        x = np.random.default_rng(0).standard_normal((100, 8))
        with pytest.raises(ValueError, match="reproducible"):
            uniformity(x, max_samples=10, rng=None)

    def test_subsampling_with_a_seed_is_reproducible(self):
        x = np.random.default_rng(0).standard_normal((100, 8))
        a = uniformity(x, max_samples=10, rng=np.random.default_rng(5))
        b = uniformity(x, max_samples=10, rng=np.random.default_rng(5))
        assert a == b

    def test_rejects_a_single_embedding(self):
        with pytest.raises(ValueError, match="at least 2"):
            uniformity(np.ones((1, 4)))


class TestNormStats:
    def test_unit_rows(self):
        stats = embedding_norm_stats(l2_normalize(
            np.random.default_rng(0).standard_normal((32, 8))
        ))
        assert stats["norm_mean"] == pytest.approx(1.0, abs=1e-9)
        assert stats["norm_std"] == pytest.approx(0.0, abs=1e-9)
        assert stats["n_zero_norm"] == 0.0

    def test_a_zero_row_is_counted_and_does_not_produce_nan(self):
        x = np.eye(4)
        x[2] = 0.0
        stats = embedding_norm_stats(x)
        assert stats["n_zero_norm"] == 1.0
        assert all(not math.isnan(v) for v in stats.values())

    def test_a_collapsed_space_has_centroid_cosine_one(self):
        v = np.random.default_rng(0).standard_normal(16)
        stats = embedding_norm_stats(np.tile(v, (50, 1)))
        assert stats["cos_to_centroid_mean"] == pytest.approx(1.0, abs=1e-9)
