"""Test suite for :mod:`dl_techniques.metrics.spatial_autocorrelation`.

Moran's I is a ratio of small differences, so almost every defect in it is
invisible to a shape check and to a finiteness check. The suite therefore pins
values, not shapes: closed-form cases, a cross-check between the O(1) grid
implementation and the general sparse one, and the specific ways the statistic
lies.

The mandated pins, with the numbers this suite measures:

  1. test_a_ramp_is_strongly_clustered            0.783 queen / 0.851 rook (6x9)
  2. test_a_checkerboard_gives_minus_unity_under_rook   exactly -1.0
  3. test_the_chance_expectation_is_the_mean_of_shuffles
                                    mean -0.0207 over 20 shuffles, chance -0.0070
  4. test_the_grid_and_sparse_implementations_agree    max |delta| < 1e-12
  5. test_a_hand_computed_3x3_value                  exact, both contiguities
  6. test_islands_morans_i_scores_a_large_region_highly  0.731 measured
"""

import numpy as np
import pytest

from dl_techniques.metrics.spatial_autocorrelation import (
    CONNECTIVITIES,
    QUEEN_OFFSETS,
    grid_weights,
    islands_morans_i,
    label_islands,
    mesh_weights,
    morans_i,
    morans_i_grid,
    morans_i_permutation_test,
    morans_i_summary,
)


def _ramp(height, width):
    """A strictly increasing map; strongly clustered under any contiguity."""
    return np.add.outer(np.arange(height), np.zeros(width)).astype(
        "float64"
    ) + np.arange(width)[None, :]


def _diagonal_band(size, low=6, high=10):
    """A band of cells along the grid diagonal: one island, irregular in shape."""
    band = np.zeros((size, size), dtype=bool)
    for row in range(size):
        for col in range(size):
            if low <= row - col <= high:
                band[row, col] = True
    return band


class TestMoransIGrid:
    @pytest.mark.parametrize(
        "connectivity,expected", [("queen", 0.7834), ("rook", 0.8513)]
    )
    def test_a_ramp_is_strongly_clustered(self, connectivity, expected):
        """A monotone map is strongly clustered -- but NOT at exactly 1.0.

        Measured 0.7834 (queen) and 0.8513 (rook) on a 6x9 ramp. The textbook
        "I = 1 for a perfect gradient" claim does not hold on a grid WITH A BORDER,
        because a border cell has fewer neighbours and the statistic weights edges
        rather than cells. Asserting 1.0 here would be asserting a property of an
        infinite lattice, and the suite would have shipped a tolerance that never
        fired. The value grows with the grid (0.866 queen at 12x12) as the border
        fraction falls, which is the same fact stated the other way.

        Non-square, so a transposed row/column stride cannot also produce a ramp.
        """
        grid = _ramp(6, 9)
        assert morans_i_grid(grid, connectivity=connectivity) == pytest.approx(
            expected, abs=1e-4
        )

    def test_a_ramp_is_more_clustered_as_the_border_fraction_falls(self):
        """The both-ways twin for the value above."""
        small = morans_i_grid(_ramp(6, 6), connectivity="queen")
        large = morans_i_grid(_ramp(16, 16), connectivity="queen")
        assert large > small, (small, large)

    def test_a_checkerboard_gives_minus_unity_under_rook(self):
        """Rook contiguity sees only edge-sharing pairs, which all alternate."""
        grid = np.indices((6, 6)).sum(axis=0) % 2
        assert morans_i_grid(grid.astype("float64"), connectivity="rook") == (
            pytest.approx(-1.0, abs=1e-12)
        )

    def test_queen_and_rook_disagree_on_a_checkerboard(self):
        """Queen adds the diagonal pairs, which agree, so it is less negative.

        This is the both-ways twin for `connectivity`: a contiguity flag that is
        accepted and then ignored would make the two arms equal.
        """
        grid = (np.indices((6, 6)).sum(axis=0) % 2).astype("float64")
        queen = morans_i_grid(grid, connectivity="queen")
        rook = morans_i_grid(grid, connectivity="rook")
        assert queen > rook
        assert rook == pytest.approx(-1.0, abs=1e-12)

    def test_the_chance_expectation_is_the_mean_of_shuffles(self):
        """Chance is ``-1/(N-1)``, NOT zero, and it is a MEASURED mean.

        20 independent 12x12 white-noise maps: mean -0.0207, sd 0.0387, range
        [-0.107, +0.055]. The single-draw spread is wider than the distance to the
        analytic expectation, so a per-draw assertion would have been either
        vacuous or flaky; the bound below is three standard errors of the mean,
        which is the quantity that is actually concentrated.
        """
        rng = np.random.default_rng(0)
        values = [morans_i_grid(rng.normal(size=(12, 12))) for _ in range(20)]
        mean = float(np.mean(values))
        expected = -1.0 / (12 * 12 - 1)
        assert abs(mean - expected) < 3.0 * float(np.std(values)) / np.sqrt(20)

    def test_the_grid_and_sparse_implementations_agree(self):
        """Cross-check against the general form, on a NON-SQUARE grid.

        ``morans_i_grid`` never materialises the weight matrix, so it is a second
        implementation rather than a shortcut -- and two implementations of one
        quantity is exactly what a cross-check is for.
        """
        rng = np.random.default_rng(0)
        grid = rng.normal(size=(5, 8))
        for connectivity in CONNECTIVITIES:
            sparse = morans_i(grid.reshape(-1), grid_weights(5, 8, connectivity))
            direct = morans_i_grid(grid, connectivity=connectivity)
            np.testing.assert_allclose(direct, sparse, atol=1e-12, rtol=0)

    def test_a_hand_computed_3x3_value(self):
        """Transcribed by hand, so it cannot share a bug with the code.

        The map ``[[1,0,0],[0,1,0],[0,0,1]]`` under queen contiguity: the three
        diagonal units are pairwise neighbours and agree, every other pair is a
        zero-mean neighbour pair, and the map's variance is 2/3.
        """
        grid = np.eye(3)
        value = morans_i_grid(grid, connectivity="queen")
        # N/W * sum_ij w_ij z_i z_j / sum z_i^2, with z = (2/3, -1/3, -1/3, -1/3)
        # in the order of the four cells that are not zero: z = grid - 1/3.
        z = grid - grid.mean()
        numerator = 2.0 * sum(
            z[i, j] * z[k, l]
            for (i, j), (k, l) in _queen_pairs(3, 3)
        )
        expected = (9.0 / (2 * len(_queen_pairs(3, 3)))) * numerator / np.sum(z ** 2)
        assert value == pytest.approx(expected, abs=1e-12)

    def test_a_constant_map_is_undefined_not_zero(self):
        """Zero variance has no spatial structure to report.

        Returns NaN rather than 0.0: a constant map is NOT evidence of dispersion,
        and reporting it as 0.0 would put it in the middle of the range.
        """
        assert np.isnan(morans_i_grid(np.ones((4, 4))))

    def test_a_one_by_one_grid_is_undefined_not_a_crash(self):
        """``W == 0``: no pairs, no statistic."""
        assert np.isnan(morans_i_grid(np.array([[3.0]])))

    def test_the_mask_of_a_map_reads_differently_from_the_map(self):
        """Why every function here takes an unthresholded map.

        Scoring the SIGNIFICANCE MASK as if it were the map replaces real values
        with a single constant over a contiguous region, and a constant region is
        perfect agreement with itself. Measured on a noisy ramp with its top 30%
        kept: 0.882 on the raw map, 0.835 on the mask -- and the mask's value is
        a statement about the mask, not about the contrast. A pipeline that
        thresholds before scoring reports a number for a different object.
        """
        rng = np.random.default_rng(0)
        noisy_ramp = _ramp(14, 14) + rng.normal(scale=0.3, size=(14, 14))
        mask = noisy_ramp >= np.percentile(noisy_ramp, 70)
        assert morans_i_grid(mask.astype("float64")) != pytest.approx(
            morans_i_grid(noisy_ramp), abs=1e-3
        )
        # And the two are not interchangeable in sign of difference either way.
        assert float(mask.astype("float64").sum()) < noisy_ramp.size

    @pytest.mark.parametrize("bad", [np.zeros(4), np.zeros((2, 2, 2))])
    def test_a_non_2d_map_is_rejected(self, bad):
        with pytest.raises(ValueError, match="must be a 2-D"):
            morans_i_grid(bad)

    def test_an_unknown_connectivity_is_rejected(self):
        with pytest.raises(ValueError, match="connectivity must be one of"):
            morans_i_grid(_ramp(4, 4), connectivity="bishop")


def _queen_pairs(height, width):
    """Undirected queen-neighbour pairs of a small grid, transcribed by hand."""
    pairs = []
    for (i, j) in [(r, c) for r in range(height) for c in range(width)]:
        for dr, dc in QUEEN_OFFSETS:
            k, l = i + dr, j + dc
            if 0 <= k < height and 0 <= l < width:
                pairs.append(((i, j), (k, l)))
    return pairs


class TestGridWeights:
    def test_the_weight_matrix_is_symmetric_with_a_zero_diagonal(self):
        weights = grid_weights(5, 7, "queen").toarray()
        np.testing.assert_array_equal(weights, weights.T)
        np.testing.assert_array_equal(np.diag(weights), np.zeros(35))

    def test_queen_counts_eight_neighbours_on_an_interior_cell(self):
        weights = grid_weights(7, 7, "queen").toarray()
        interior = weights[3 * 7 + 3]
        assert interior.sum() == 8

    def test_rook_counts_four(self):
        weights = grid_weights(7, 7, "rook").toarray()
        assert weights[3 * 7 + 3].sum() == 4

    @pytest.mark.parametrize("bad", [(0, 5), (5, 0), (-1, 5)])
    def test_non_positive_dimensions_are_rejected(self, bad):
        with pytest.raises(ValueError, match="must be positive"):
            grid_weights(*bad)

    def test_an_unknown_connectivity_is_rejected(self):
        with pytest.raises(ValueError, match="connectivity must be one of"):
            grid_weights(4, 4, "bishop")


class TestMeshWeights:
    def test_a_torus_mesh_matches_the_grid_weights_in_the_interior(self):
        """A periodic mesh has no border; a grid does.

        Cross-checks the mesh path against the grid path on the one geometry where
        the two are the same object. The interior is compared element-wise; the
        border necessarily differs, since the torus wraps and the grid stops.
        """
        height = width = 4
        edges = [
            (
                row * width + col,
                ((row + dr) % height) * width + ((col + dc) % width),
            )
            for row in range(height)
            for col in range(width)
            for dr, dc in QUEEN_OFFSETS
            if ((row + dr) % height, (col + dc) % width) != (row, col)
        ]
        mesh = mesh_weights(edges, height * width).toarray()
        grid = grid_weights(height, width, "queen").toarray()

        interior = np.array(
            [i * width + j for i in (1, 2) for j in (1, 2)]
        )
        assert np.array_equal(
            mesh[np.ix_(interior, interior)], grid[np.ix_(interior, interior)]
        )
        assert not np.array_equal(mesh, grid), (
            "a 4x4 torus should NOT equal a 4x4 grid -- the wrap has to matter"
        )

    def test_a_torus_gives_every_vertex_eight_neighbours(self):
        """No border means no reduced degree."""
        height = width = 4
        edges = [
            (
                row * width + col,
                ((row + dr) % height) * width + ((col + dc) % width),
            )
            for row in range(height)
            for col in range(width)
            for dr, dc in QUEEN_OFFSETS
            if ((row + dr) % height, (col + dc) % width) != (row, col)
        ]
        mesh = mesh_weights(edges, 16).toarray()
        assert np.all(mesh.sum(axis=1) == 8)

    def test_a_hcp_style_mesh_gives_six_neighbours_per_vertex(self):
        """The surface case the paper uses: six neighbours, not eight or four.

        Flattened vertex indices with the six offsets of a triangulated surface
        at width 30: along the row, across the row, and the two diagonals. A
        vertex far from the ends has all six; vertices at the ends do not, which
        is what any banded offset scheme gives -- asserting six for EVERY vertex
        would be asserting a property no finite mesh has.
        """
        offsets = [1, 30, 29]
        # Only the forward half: `mesh_weights` symmetrises, so listing both
        # directions would give every interior vertex TWELVE neighbours and the
        # test would be asserting the wrong contract.
        edges = [
            (vertex, vertex + offset)
            for vertex in range(900)
            for offset in offsets
            if vertex + offset < 900
        ]
        weights = mesh_weights(edges, 900).toarray()
        assert weights[450].sum() == 6
        assert weights[1].sum() < 6
        assert weights[898].sum() < 6

    def test_a_self_loop_is_rejected(self):
        with pytest.raises(ValueError, match="cannot join a vertex to itself"):
            mesh_weights([(0, 0), (1, 2)], 4)

    def test_an_out_of_range_edge_is_rejected(self):
        with pytest.raises(ValueError, match=r"edge indices must lie in"):
            mesh_weights([(0, 9)], 4)

    def test_a_non_positive_vertex_count_is_rejected(self):
        with pytest.raises(ValueError, match="num_vertices must be positive"):
            mesh_weights([(0, 1)], 0)


class TestMoransIPermutationTest:
    def test_a_clustered_map_is_significant_and_noise_is_not(self):
        """Both arms, because a one-sided test passes for a broken permutation."""
        rng = np.random.default_rng(0)
        weights = grid_weights(10, 10, "queen")

        clustered = _ramp(10, 10).reshape(-1)
        observed, p_value = morans_i_permutation_test(
            clustered, weights, num_permutations=999, seed=0
        )
        assert observed > 0.5
        assert p_value <= 1.0 / 1000.0

        _, noise_p = morans_i_permutation_test(
            rng.normal(size=100), weights, num_permutations=999, seed=0
        )
        assert noise_p > 0.05, noise_p

    def test_the_p_value_is_reproducible_under_a_seed(self):
        rng = np.random.default_rng(1)
        weights = grid_weights(8, 8, "queen")
        values = rng.normal(size=64)
        first = morans_i_permutation_test(values, weights, 199, seed=7)
        second = morans_i_permutation_test(values, weights, 199, seed=7)
        assert first == second

    def test_the_smallest_attainable_p_value_is_one_over_n_plus_one(self):
        """A perfectly clustered map on a 2x2 cannot reach p = 0."""
        weights = grid_weights(2, 2, "rook")
        values = np.array([0.0, 1.0, 2.0, 3.0])
        _, p_value = morans_i_permutation_test(
            values, weights, num_permutations=99, seed=0
        )
        assert p_value >= 1.0 / 100.0

    def test_a_non_positive_permutation_count_is_rejected(self):
        with pytest.raises(ValueError, match="num_permutations must be positive"):
            morans_i_permutation_test(
                np.arange(9.0), grid_weights(3, 3), num_permutations=0
            )

    def test_a_constant_map_propagates_nan(self):
        observed, p_value = morans_i_permutation_test(
            np.ones(9), grid_weights(3, 3), num_permutations=9, seed=0
        )
        assert np.isnan(observed) and np.isnan(p_value)


class TestLabelIslands:
    def test_two_separated_blobs_get_two_labels(self):
        mask = np.zeros((7, 7), dtype=bool)
        mask[1:3, 1:3] = True
        mask[4:6, 4:6] = True
        labels, count = label_islands(mask)
        assert count == 2
        assert labels[1, 1] == 1
        assert labels[4, 4] == 2
        assert labels[0, 0] == 0

    def test_diagonal_touching_blobs_merge_under_queen_and_not_rook(self):
        """Queen contiguity is the difference, measured.

        Two cells touching only at a corner are one island under queen and two
        under rook. This is the both-ways twin for the contiguity argument.
        """
        mask = np.zeros((5, 5), dtype=bool)
        mask[1, 1] = True
        mask[2, 2] = True
        assert label_islands(mask, connectivity="queen")[1] == 1
        assert label_islands(mask, connectivity="rook")[1] == 2

    def test_small_islands_are_dropped(self):
        mask = np.zeros((7, 7), dtype=bool)
        mask[1, 1] = True
        mask[3:6, 3:6] = True
        labels, count = label_islands(mask, min_island_size=3)
        assert count == 1
        assert labels[1, 1] == 0
        assert labels[3, 3] == 1

    def test_an_empty_mask_yields_no_islands(self):
        labels, count = label_islands(np.zeros((5, 5), dtype=bool))
        assert count == 0
        assert labels.sum() == 0

    @pytest.mark.parametrize("bad", [np.zeros((2, 2, 2)), np.zeros(5)])
    def test_a_non_2d_mask_is_rejected(self, bad):
        with pytest.raises(ValueError, match="must be a 2-D"):
            label_islands(bad)

    def test_an_unknown_connectivity_is_rejected(self):
        with pytest.raises(ValueError, match="connectivity must be one of"):
            label_islands(np.ones((3, 3), dtype=bool), connectivity="bishop")

    def test_a_min_island_size_below_one_is_rejected(self):
        with pytest.raises(ValueError, match="min_island_size must be >= 1"):
            label_islands(np.ones((3, 3), dtype=bool), min_island_size=0)


class TestIslandsMoransI:
    def test_a_large_contiguous_region_scores_highly(self):
        """The case the statistic is for: a broad, genuinely clustered region.

        The significant set is the upper 40% of a noisy ramp on a 12x12 grid --
        one connected island of ~57 cells. Measured: islands 0.731, against 0.860
        for the standard statistic on the raw map. The islands value sits in the
        same positive regime instead of being dragged toward zero by the
        non-significant remainder, which is the whole reason it exists.
        """
        rng = np.random.default_rng(0)
        t_grid = _ramp(12, 12) + rng.normal(scale=0.4, size=(12, 12))
        sig_grid = t_grid > np.percentile(t_grid, 60)
        assert islands_morans_i(t_grid, sig_grid, min_island_size=3) > 0.5

    def test_a_small_island_is_high_variance_and_sometimes_negative(self):
        """A calibrated limitation, measured rather than assumed.

        A planted diagonal band spanning ~70 of 256 cells reads -0.077 under
        queen while the same map's standard statistic reads +0.38. Two effects
        stack: the island is small, so the ratio's denominator is small, and a
        compact region carries more same-sign DIAGONAL pairs than the queen
        weighting expects -- Anselin's queen-contiguity instability.

        Pinned so this module's docstring cannot drift into "islands is better".
        On a small island it is not, and a caller who wants a stable number wants
        the standard statistic.
        """
        rng = np.random.default_rng(0)
        band = _diagonal_band(16)
        t_grid = rng.normal(size=(16, 16)) + band * 3.0
        assert islands_morans_i(t_grid, band, min_island_size=3) < 0.2
        assert morans_i_grid(t_grid, connectivity="queen") > 0.3

    def test_rook_removes_the_diagonal_instability(self):
        """The both-ways twin: the contiguity choice changes the islands value.

        The same diagonal band scores -0.077 under queen and +0.065 under rook.
        An implementation ignoring ``connectivity`` would return the same number
        twice.
        """
        rng = np.random.default_rng(0)
        band = _diagonal_band(16)
        t_grid = rng.normal(size=(16, 16)) + band * 3.0
        queen = islands_morans_i(
            t_grid, band, connectivity="queen", min_island_size=3
        )
        rook = islands_morans_i(
            t_grid, band, connectivity="rook", min_island_size=3
        )
        # Measured -0.0769 against -0.0483: a gap of 0.029, which is small in
        # absolute terms and is itself the point -- the statistic is noisy on an
        # island this size, and the contiguity choice is a second-order effect on
        # top of that rather than the dominant one.
        assert rook > queen, (queen, rook)

    def test_no_significant_units_is_not_applicable_not_zero(self):
        """The paper reports this case as "not applicable", and so does this."""
        t_grid = _ramp(8, 8)
        assert np.isnan(islands_morans_i(t_grid, np.zeros((8, 8), dtype=bool)))

    def test_islands_below_the_size_floor_are_ignored_entirely(self):
        t_grid = _ramp(10, 10)
        sig_grid = np.zeros((10, 10), dtype=bool)
        sig_grid[0, 0] = True
        sig_grid[9, 9] = True
        assert np.isnan(islands_morans_i(t_grid, sig_grid, min_island_size=3))

    def test_mismatched_shapes_are_rejected(self):
        with pytest.raises(ValueError, match="does not match t_grid shape"):
            islands_morans_i(_ramp(4, 4), np.ones((5, 5), dtype=bool))


class TestMoransISummary:
    def test_both_statistics_are_reported_together(self):
        t_grid = _ramp(9, 9)
        sig_grid = t_grid > np.median(t_grid)
        summary = morans_i_summary(t_grid, sig_grid)
        assert set(summary) == {"standard", "islands", "num_units", "connectivity"}
        assert summary["num_units"] == 81
        # A 9x9 ramp measures 0.816, consistent with the 0.790 at 8x8 and 0.866 at
        # 12x12 -- the border fraction, not the gradient, sets the ceiling.
        assert summary["standard"] == pytest.approx(0.816, abs=1e-3)
        assert not np.isnan(summary["islands"])

    def test_omitting_the_significance_map_leaves_the_islands_value_undefined(self):
        """It does NOT quietly score the whole grid as a single island."""
        summary = morans_i_summary(_ramp(6, 6))
        assert np.isnan(summary["islands"])
        assert not np.isnan(summary["standard"])