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

import hashlib

import numpy as np
import pytest
from scipy import ndimage

from dl_techniques.layers.regularization.spatial_smoothness import SpatialLayout
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

    @pytest.mark.parametrize("bad", [np.zeros(4), np.zeros((2, 2, 2, 2))])
    def test_a_non_2d_map_is_rejected(self, bad):
        """Rewritten in plan 972a9ac5 step 2 (D-003). It used to parametrize
        ``np.zeros((2, 2, 2))`` as a map that must be refused; a 3-D map is now
        scored (``TestMoransIGrid3D``), so the refused ranks are the ones the
        grid functions still do not support: 1-D and 4-D. The message still names
        the rank form."""
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

    @pytest.mark.parametrize("bad", [np.zeros((2, 2, 2, 2)), np.zeros(5)])
    def test_a_non_2d_mask_is_rejected(self, bad):
        """Rewritten in plan 972a9ac5 step 2 (D-003): ``np.zeros((2, 2, 2))`` is
        no longer an example of a refused mask, because ``label_islands`` labels
        a 3-D mask (``TestLabelIslands3D``). 4-D and 1-D are still refused, by
        name."""
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
    def test_all_three_statistics_are_reported_together(self):
        t_grid = _ramp(9, 9)
        sig_grid = t_grid > np.median(t_grid)
        summary = morans_i_summary(t_grid, sig_grid)
        assert set(summary) == {
            "standard",
            "islands",
            "permutation_p",
            "num_permutations",
            "num_units",
            "connectivity",
        }
        assert summary["num_units"] == 81
        # A 9x9 ramp measures 0.816, consistent with the 0.790 at 8x8 and 0.866 at
        # 12x12 -- the border fraction, not the gradient, sets the ceiling.
        assert summary["standard"] == pytest.approx(0.816, abs=1e-3)
        assert not np.isnan(summary["islands"])

    def test_the_permutation_statistic_is_skipped_unless_asked_for(self):
        """Off by default, because it costs a shuffle per permutation.

        A 24-tap, 2-arm run at the 9999 default is ~480k Moran's I
        recomputations, so silently enabling it would make the default analysis
        hundreds of times slower for a statistic most runs never read.
        """
        summary = morans_i_summary(_ramp(6, 6))
        assert summary["num_permutations"] is None
        assert np.isnan(summary["permutation_p"])

    def test_a_smooth_map_is_far_from_spatial_randomness(self):
        """The whole point of the permutation p-value: the two arms differ.

        Without this, reporting a p-value would be reporting a number nobody has
        checked is capable of being small.
        """
        t_grid = _ramp(9, 9)
        summary = morans_i_summary(
            t_grid, num_permutations=199, seed=7
        )
        assert summary["num_permutations"] == 199
        assert summary["permutation_p"] == pytest.approx(1 / 200), (
            f"a linear ramp has the most clustered arrangement possible, so its "
            f"p-value should sit at the floor; got {summary['permutation_p']}"
        )

    def test_noise_is_not_distinguishable_from_randomness(self):
        """The null arm: shuffled noise cannot beat itself consistently."""
        generator = np.random.default_rng(11)
        values = generator.normal(size=(9, 9))
        summary = morans_i_summary(
            values, num_permutations=199, seed=3
        )
        assert summary["permutation_p"] > 0.05, (
            f"random data scored p={summary['permutation_p']}, which would make "
            f"the statistic look informative about nothing"
        )

    def test_the_permutation_p_value_is_reproducible_from_its_seed(self):
        """Two calls with one seed must agree exactly.

        `hash()` would break this -- it is salted per process -- which is why the
        seed is a plain integer mix rather than a digest of the tap's path.
        """
        t_grid = _ramp(8, 8)
        first = morans_i_summary(t_grid, num_permutations=99, seed=5)
        second = morans_i_summary(t_grid, num_permutations=99, seed=5)
        assert first["permutation_p"] == second["permutation_p"]
        # A different count of permutations must move the p-value's floor, which
        # is the way to show the seed reaches the statistic at all rather than
        # being accepted and discarded.
        fewer = morans_i_summary(t_grid, num_permutations=49, seed=5)
        assert fewer["permutation_p"] >= first["permutation_p"] - 0.05

    def test_a_non_positive_permutation_count_is_rejected(self):
        for bad in (0, -1):
            with pytest.raises(ValueError, match="must be positive"):
                morans_i_summary(_ramp(4, 4), num_permutations=bad)

    def test_a_constant_map_reports_an_undefined_p_value(self):
        """Moran's I is nan on a constant map, and nan must propagate, not zero."""
        summary = morans_i_summary(
            np.ones((6, 6)), num_permutations=19, seed=1
        )
        assert np.isnan(summary["standard"])
        assert np.isnan(summary["permutation_p"])

    def test_omitting_the_significance_map_leaves_the_islands_value_undefined(self):
        """It does NOT quietly score the whole grid as a single island."""
        summary = morans_i_summary(_ramp(6, 6))
        assert np.isnan(summary["islands"])
        assert not np.isnan(summary["standard"])

# =====================================================================
# plan-2026-10-23T190557-972a9ac5, step 1: pins and RED tests
#
# Two kinds of test follow. The ``*GoldenPins`` classes record what the 2-D
# functions return TODAY (``float.hex`` literals and CSR digests measured at
# commit 9a1a446ec) and must stay green through every later step: the N-d
# rewrite is allowed to add rank 3, it is not allowed to move a 2-D bit. The
# ``*3D`` classes are the oracles for that rank; they were RED when written,
# because the functions refused any map that is not 2-D, and each one compares
# the code with something that does not import the code: a triple loop, scipy's
# own labelling, a closed form.
# =====================================================================

def _digest(array):
    """First 16 hex digits of the SHA-256 of an array's bytes (same environment pin)."""
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()[:16]


def _golden_island_map():
    """A 14x14 map with a 20-cell, a 9-cell, a 2-cell, a diagonal pair and a lone cell."""
    significant = np.zeros((14, 14), dtype=bool)
    significant[1:6, 1:5] = True
    significant[8:11, 2:5] = True
    significant[2, 9] = significant[2, 10] = True
    significant[7, 8] = significant[8, 9] = True
    significant[12, 12] = True
    values = np.random.default_rng(21).normal(size=(14, 14)) + 2.0 * significant
    return values, significant


def _golden_layout_map():
    """A 12x12 map put on the grid through a permuted layout, and its mask."""
    layout = SpatialLayout(144, seed=3)
    per_unit = np.random.default_rng(22).normal(size=144)
    return layout.to_grid(per_unit), layout.to_grid(per_unit > 0.6)


class TestMoransIGoldenPins:
    """The 2-D statistic, bit for bit. Measured on the unmodified code."""

    #: ``(shape, seed, sigma)`` -> ``{connectivity: float.hex}``. ``sigma`` None is
    #: white noise; a number smooths it first, so the pins also cover values far
    #: from chance, where a lost term would not hide inside rounding.
    GOLDEN = {
        ((28, 28), 11, None): {
            "queen": "-0x1.c5f4b1fcc048ap-7", "rook": "-0x1.034a80fb89d61p-7",
        },
        ((5, 8), 12, None): {
            "queen": "-0x1.bb53117104761p-7", "rook": "-0x1.796addd071731p-4",
        },
        ((28, 28), 14, 2.0): {
            "queen": "0x1.c6ec99c385710p-1", "rook": "0x1.d896f1bab5f50p-1",
        },
        ((5, 8), 15, 1.0): {
            "queen": "0x1.04d9c11e20e0fp-1", "rook": "0x1.3c0cd05913dd2p-1",
        },
    }

    @staticmethod
    def _field(shape, seed, sigma):
        field = np.random.default_rng(seed).normal(size=shape)
        return field if sigma is None else ndimage.gaussian_filter(field, sigma)

    @pytest.mark.parametrize("key", list(GOLDEN), ids=lambda k: f"{k[0]}-seed{k[1]}")
    @pytest.mark.parametrize("connectivity", ["queen", "rook"])
    def test_morans_i_grid_is_bit_identical(self, key, connectivity):
        value = morans_i_grid(self._field(*key), connectivity=connectivity)
        assert float(value).hex() == self.GOLDEN[key][connectivity], (
            f"morans_i_grid moved on a 2-D map ({key}, {connectivity}): "
            f"{float(value).hex()}"
        )

    #: ``(map, significance)`` kind -> connectivity -> (standard, islands) hex.
    SUMMARY = {
        ("plain", "queen"): ("0x1.a4216139f2444p-3", "0x1.4d314d485e7e8p-6"),
        ("plain", "rook"): ("0x1.033912f709ac1p-2", "0x1.afffc6c53b6a9p-3"),
        ("layout", "queen"): ("-0x1.389b7723507fap-5", "-0x1.2298a90310c10p-2"),
        ("layout", "rook"): ("-0x1.2c952fc6c0a9ep-5", "-0x1.afc80adf5cc9bp-2"),
    }

    @staticmethod
    def _map(kind):
        return _golden_island_map() if kind == "plain" else _golden_layout_map()

    @pytest.mark.parametrize("kind,connectivity", list(SUMMARY))
    def test_morans_i_summary_is_bit_identical_with_and_without_a_mask(
        self, kind, connectivity
    ):
        """``plain`` is a map with planted islands, ``layout`` goes through
        ``SpatialLayout.to_grid``; each is scored with a mask, then without one
        (where the islands value must be ``nan``, not a whole-grid island)."""
        values, mask = self._map(kind)
        standard, islands = self.SUMMARY[(kind, connectivity)]

        with_mask = morans_i_summary(values, mask, connectivity=connectivity)
        assert set(with_mask) == {
            "standard", "islands", "permutation_p", "num_permutations",
            "num_units", "connectivity",
        }
        assert float(with_mask["standard"]).hex() == standard
        assert float(with_mask["islands"]).hex() == islands
        assert np.isnan(with_mask["permutation_p"])
        assert with_mask["num_permutations"] is None
        assert with_mask["num_units"] == values.size
        assert with_mask["connectivity"] == connectivity

        without_mask = morans_i_summary(values, None, connectivity=connectivity)
        assert float(without_mask["standard"]).hex() == standard
        assert np.isnan(without_mask["islands"])

    @pytest.mark.parametrize(
        "kind,expected",
        [("plain", "0x1.47ae147ae147bp-7"), ("layout", "0x1.947ae147ae148p-1")],
    )
    def test_the_permutation_p_value_is_bit_identical(self, kind, expected):
        values, mask = self._map(kind)
        summary = morans_i_summary(values, mask, num_permutations=99, seed=5)
        assert float(summary["permutation_p"]).hex() == expected
        assert summary["num_permutations"] == 99

    #: ``(connectivity, min_island_size)`` -> ``float.hex`` of the island mean.
    ISLANDS = {
        ("queen", 1): "-0x1.f5967595bd0c0p-2",
        ("queen", 3): "0x1.4d314d485e7e8p-6",
        ("queen", 10): "0x1.012769585dd4ep-2",
        ("rook", 1): "-0x1.8aaad0d1d863bp-3",
        ("rook", 3): "0x1.afffc6c53b6a9p-3",
        ("rook", 10): "0x1.41b0157c3fb1cp-2",
    }

    @pytest.mark.parametrize("key", list(ISLANDS), ids=lambda k: f"{k[0]}-min{k[1]}")
    def test_islands_morans_i_is_bit_identical(self, key):
        values, mask = _golden_island_map()
        connectivity, minimum = key
        value = islands_morans_i(
            values, mask, connectivity=connectivity, min_island_size=minimum
        )
        assert float(value).hex() == self.ISLANDS[key]

    #: ``(connectivity, min_island_size)`` -> ``(count, digest of the int64 labels)``.
    LABELS = {
        ("queen", 1): (5, "432c951917e662f7"),
        ("queen", 3): (2, "a9041cc11d334002"),
        ("queen", 10): (1, "3e27b20019bd2f35"),
        ("rook", 1): (6, "e502a5a8de96b98e"),
        ("rook", 3): (2, "a9041cc11d334002"),
        ("rook", 10): (1, "3e27b20019bd2f35"),
    }

    @pytest.mark.parametrize("key", list(LABELS), ids=lambda k: f"{k[0]}-min{k[1]}")
    def test_label_islands_is_bit_identical(self, key):
        _, mask = _golden_island_map()
        connectivity, minimum = key
        labels, count = label_islands(
            mask, connectivity=connectivity, min_island_size=minimum
        )
        assert (count, _digest(labels)) == self.LABELS[key]
        assert labels.dtype == np.int64


class TestGridWeightsGoldenPins:
    """The 2-D neighbour matrix, array for array."""

    def test_a_2x3_queen_grid_is_these_exact_csr_arrays(self):
        weights = grid_weights(2, 3, "queen")
        assert weights.indptr.tolist() == [0, 3, 8, 11, 14, 19, 22]
        assert weights.indices.tolist() == [
            1, 3, 4, 0, 2, 3, 4, 5, 1, 4, 5, 0, 1, 4, 0, 1, 2, 3, 5, 1, 2, 4,
        ]
        assert weights.data.tolist() == [1.0] * 22

    def test_a_3x4_rook_grid_is_these_exact_csr_arrays(self):
        weights = grid_weights(3, 4, "rook")
        assert weights.indptr.tolist() == [
            0, 2, 5, 8, 10, 13, 17, 21, 24, 26, 29, 32, 34,
        ]
        assert weights.indices.tolist() == [
            1, 4, 0, 2, 5, 1, 3, 6, 2, 7, 0, 5, 8, 1, 4, 6, 9, 2, 5, 7, 10,
            3, 6, 11, 4, 9, 5, 8, 10, 6, 9, 11, 7, 10,
        ]
        assert weights.data.tolist() == [1.0] * 34

    #: ``(height, width, connectivity)`` -> ``(nnz, indptr, indices, data)`` digests.
    DIGESTS = {
        (5, 8, "queen"): (246, "e5bbd35c468af9ee", "5fb2db689c1526bc", "3db3077cd98605c4"),
        (5, 8, "rook"): (134, "f0dff6672974a86b", "eceaa2dd424e205e", "890713daabab3597"),
        (28, 28, "queen"): (5940, "b56eec9e38183de4", "ab758ce5d4bcc6aa", "f59fad8483ab0b14"),
        (28, 28, "rook"): (3024, "3b9f4931474fffc3", "1b341333796a609c", "a03c16be4041fc25"),
    }

    @pytest.mark.parametrize("key", list(DIGESTS), ids=lambda k: f"{k[0]}x{k[1]}-{k[2]}")
    def test_larger_grids_keep_their_csr_digests(self, key):
        height, width, connectivity = key
        weights = grid_weights(height, width, connectivity)
        nnz, indptr, indices, data = self.DIGESTS[key]
        assert weights.nnz == nnz
        assert (weights.indptr.dtype, weights.indices.dtype, weights.data.dtype) == (
            np.int32, np.int32, np.float64,
        )
        assert weights.has_sorted_indices
        assert (_digest(weights.indptr), _digest(weights.indices),
                _digest(weights.data)) == (indptr, indices, data)


# ---------------------------------------------------------------------
# rank 3: oracles that do not import the implementation
# ---------------------------------------------------------------------


def _brute_force_morans_i(grid, connectivity):
    """Moran's I by an explicit loop over every cell and every neighbour offset.

    Binary weights, normaliser ``W`` = total number of ordered neighbour pairs
    (not row-standardised), as in the module docstring. Shares nothing with the
    implementation: no offset table, no slicing, no sparse matrix.
    """
    grid = np.asarray(grid, dtype="float64")
    mean = grid.mean()
    denominator = float(((grid - mean) ** 2).sum())
    numerator = 0.0
    pairs = 0
    for cell in np.ndindex(*grid.shape):
        for step in np.ndindex(*([3] * grid.ndim)):
            offset = tuple(s - 1 for s in step)
            if not any(offset):
                continue
            if connectivity == "rook" and sum(abs(o) for o in offset) != 1:
                continue
            other = tuple(c + o for c, o in zip(cell, offset))
            if all(0 <= x < n for x, n in zip(other, grid.shape)):
                numerator += (grid[cell] - mean) * (grid[other] - mean)
                pairs += 1
    return grid.size / pairs * numerator / denominator


def _smooth_volume(shape, seed, sigma=1.0):
    return ndimage.gaussian_filter(
        np.random.default_rng(seed).normal(size=shape), sigma
    )


_SHAPES_3D = [(5, 6, 7), (10, 10, 10), (7, 8, 14)]


class TestMoransIGrid3D:
    @pytest.mark.parametrize("shape", _SHAPES_3D, ids=str)
    @pytest.mark.parametrize("connectivity", ["queen", "rook"])
    def test_the_grid_form_matches_a_triple_loop(self, shape, connectivity):
        """Queen is the 26-neighbour cube, rook the 6 face neighbours."""
        volume = _smooth_volume(shape, seed=3)
        assert morans_i_grid(volume, connectivity=connectivity) == pytest.approx(
            _brute_force_morans_i(volume, connectivity), abs=1e-12
        )

    @pytest.mark.parametrize("shape", _SHAPES_3D, ids=str)
    @pytest.mark.parametrize("connectivity", ["queen", "rook"])
    def test_the_sparse_form_on_the_nd_weights_agrees_with_the_loop(
        self, shape, connectivity
    ):
        from dl_techniques.metrics.spatial_autocorrelation import grid_weights_nd

        volume = _smooth_volume(shape, seed=4)
        sparse = morans_i(volume.ravel(), grid_weights_nd(shape, connectivity))
        assert sparse == pytest.approx(
            _brute_force_morans_i(volume, connectivity), abs=1e-12
        )

    @pytest.mark.parametrize("seed", range(5))
    def test_a_2x2x2_map_is_minus_one_seventh_under_queen(self, seed):
        """Every cell of a 2x2x2 cube touches the other seven, so the weights are
        the complete graph and I must be ``-1/(N-1) = -1/7`` for ANY
        non-constant map (the chance value, reached exactly)."""
        cube = np.random.default_rng(seed).normal(size=(2, 2, 2))
        assert morans_i_grid(cube, connectivity="queen") == pytest.approx(
            -1.0 / 7.0, abs=1e-12
        )

    def test_a_3d_checkerboard_is_minus_one_under_rook(self):
        """Every face neighbour of a cell has the opposite parity."""
        board = (np.indices((6, 6, 6)).sum(axis=0) % 2).astype("float64")
        assert morans_i_grid(board, connectivity="rook") == pytest.approx(
            -1.0, abs=1e-12
        )

    def test_queen_and_rook_disagree_on_a_3d_checkerboard(self):
        """Queen adds the edge and corner neighbours, which mostly agree in
        parity: measured -0.07216 (96-cell sum, 6x6x6). The both-ways twin of
        the rook value above."""
        board = (np.indices((6, 6, 6)).sum(axis=0) % 2).astype("float64")
        assert morans_i_grid(board, connectivity="queen") == pytest.approx(
            -0.0721649, abs=1e-6
        )

    def test_a_ramp_on_a_non_cubic_volume_has_these_values(self):
        """MEASURED on 5x6x7: queen 0.74101, rook 0.86873. A ramp over the sum of
        the three indices is symmetric in the axes, so this pins the neighbour
        counts rather than the axis order (the (7, 8, 14) tests do that)."""
        ramp = np.add.outer(
            np.add.outer(np.arange(5), np.arange(6)), np.arange(7)
        ).astype("float64")
        assert morans_i_grid(ramp, "queen") == pytest.approx(0.7410051, abs=1e-6)
        assert morans_i_grid(ramp, "rook") == pytest.approx(0.8687300, abs=1e-6)

    def test_a_constant_volume_is_undefined(self):
        assert np.isnan(morans_i_grid(np.ones((3, 3, 3))))

    def test_a_3d_map_is_accepted(self):
        """Half of the rank guard: 3 is allowed (it was refused before)."""
        assert np.isfinite(morans_i_grid(_smooth_volume((4, 4, 4), seed=1)))

    @pytest.mark.parametrize("bad", [np.zeros(4), np.zeros((2, 2, 2, 2))], ids=["1D", "4D"])
    def test_ranks_one_and_four_are_still_refused_by_name(self, bad):
        """The other half, true before and after: the message keeps naming the
        (height, width) form, and the offending shape."""
        with pytest.raises(ValueError, match=r"2-D \(height, width\)") as caught:
            morans_i_grid(bad)
        assert str(bad.shape) in str(caught.value)


class TestGridWeightsNd:
    @staticmethod
    def _degrees(shape, connectivity):
        """Closed-form neighbour count of every cell, built per axis.

        Along one axis a cell at index ``i`` of ``n`` can step to ``i-1``, ``i``
        and ``i+1`` where they exist, so the queen count is the product of those
        per-axis choices minus the cell itself; rook counts only the single-axis
        moves.
        """
        per_axis = []
        for n in shape:
            index = np.arange(n)
            per_axis.append(1 + (index > 0) + (index < n - 1))
        if connectivity == "queen":
            total = np.ones(shape, dtype="int64")
            for axis, choices in enumerate(per_axis):
                view = [1] * len(shape)
                view[axis] = shape[axis]
                total = total * choices.reshape(view)
            return (total - 1).ravel()
        total = np.zeros(shape, dtype="int64")
        for axis, choices in enumerate(per_axis):
            view = [1] * len(shape)
            view[axis] = shape[axis]
            total = total + (choices.reshape(view) - 1)
        return total.ravel()

    @pytest.mark.parametrize("shape", _SHAPES_3D, ids=str)
    @pytest.mark.parametrize("connectivity", ["queen", "rook"])
    def test_every_cell_has_the_closed_form_degree(self, shape, connectivity):
        from dl_techniques.metrics.spatial_autocorrelation import grid_weights_nd

        weights = grid_weights_nd(shape, connectivity)
        np.testing.assert_array_equal(
            np.asarray(weights.sum(axis=1)).ravel(),
            self._degrees(shape, connectivity),
        )

    def test_a_10_cube_has_the_counted_degree_histogram(self):
        """Queen: 8 corners of 7, 96 edge cells of 11, 384 face cells of 17 and
        512 interior cells of 26 (20 952 nonzeros); rook: 3 / 4 / 5 / 6."""
        from dl_techniques.metrics.spatial_autocorrelation import grid_weights_nd

        queen = grid_weights_nd((10, 10, 10), "queen")
        degrees, counts = np.unique(
            np.asarray(queen.sum(axis=1)).ravel().astype(int), return_counts=True
        )
        assert dict(zip(degrees.tolist(), counts.tolist())) == {
            7: 8, 11: 96, 17: 384, 26: 512,
        }
        assert queen.nnz == 20952
        rook = grid_weights_nd((10, 10, 10), "rook")
        degrees, counts = np.unique(
            np.asarray(rook.sum(axis=1)).ravel().astype(int), return_counts=True
        )
        assert dict(zip(degrees.tolist(), counts.tolist())) == {
            3: 8, 4: 96, 5: 384, 6: 512,
        }
        assert rook.nnz == 5400

    @pytest.mark.parametrize("connectivity", ["queen", "rook"])
    def test_the_matrix_is_binary_symmetric_with_a_zero_diagonal(self, connectivity):
        from dl_techniques.metrics.spatial_autocorrelation import grid_weights_nd

        dense = grid_weights_nd((4, 5, 3), connectivity).toarray()
        np.testing.assert_array_equal(dense, dense.T)
        np.testing.assert_array_equal(np.diag(dense), np.zeros(60))
        assert set(np.unique(dense)) == {0.0, 1.0}

    @pytest.mark.parametrize("shape", [(3, 4), (5, 8), (28, 28)], ids=str)
    @pytest.mark.parametrize("connectivity", ["queen", "rook"])
    def test_a_2d_shape_reproduces_grid_weights_array_for_array(
        self, shape, connectivity
    ):
        """The bridge that lets the 2-D path stay bit-identical: the N-d builder
        on ``(h, w)`` is ``grid_weights(h, w)``, compared on the raw CSR arrays
        (not only on ``!=``), because the summation order of ``W @ x`` follows
        the index order inside a row."""
        from dl_techniques.metrics.spatial_autocorrelation import grid_weights_nd

        old = grid_weights(*shape, connectivity=connectivity)
        new = grid_weights_nd(shape, connectivity)
        np.testing.assert_array_equal(new.indptr, old.indptr)
        np.testing.assert_array_equal(new.indices, old.indices)
        np.testing.assert_array_equal(new.data, old.data)

    def test_bad_arguments_are_named(self):
        from dl_techniques.metrics.spatial_autocorrelation import grid_weights_nd

        with pytest.raises(ValueError, match="connectivity must be one of"):
            grid_weights_nd((3, 3, 3), "bishop")
        with pytest.raises(ValueError, match="must be positive"):
            grid_weights_nd((3, 0, 3), "queen")


def _scipy_labels(mask, connectivity):
    """``scipy.ndimage.label`` with the structure written out, not imported."""
    rank = mask.ndim
    structure = ndimage.generate_binary_structure(
        rank, rank if connectivity == "queen" else 1
    )
    return ndimage.label(mask, structure=structure)


class TestLabelIslands3D:
    @pytest.mark.parametrize("shape", [(10, 10, 10), (7, 8, 14)], ids=str)
    @pytest.mark.parametrize("connectivity", ["queen", "rook"])
    def test_islands_are_scipys_components(self, shape, connectivity):
        mask = _smooth_volume(shape, seed=5, sigma=1.0) > 0.12
        reference, count = _scipy_labels(mask, connectivity)
        assert count >= 3, "the fixture must have several islands to mean anything"
        labels, found = label_islands(mask, connectivity=connectivity)
        assert found == count
        np.testing.assert_array_equal(labels, reference)

    def test_queen_merges_a_pair_that_touches_only_at_a_corner(self):
        """The both-ways twin: a rook structure used for queen would keep two."""
        mask = np.zeros((4, 4, 4), dtype=bool)
        mask[1, 1, 1] = mask[2, 2, 2] = True
        assert label_islands(mask, "queen")[1] == 1
        assert label_islands(mask, "rook")[1] == 2

    def test_small_islands_are_dropped_and_the_rest_renumbered(self):
        mask = np.zeros((6, 6, 6), dtype=bool)
        mask[0, 0, 0] = True                      # size 1
        mask[2:4, 2:4, 2:4] = True                # size 8
        mask[5, 5, 4:6] = True                    # size 2
        labels, count = label_islands(mask, "queen", min_island_size=2)
        assert count == 2
        assert sorted(np.unique(labels).tolist()) == [0, 1, 2]
        assert labels[0, 0, 0] == 0
        assert int((labels == 1).sum()) == 8 and int((labels == 2).sum()) == 2


def _islands_oracle(values, mask, connectivity, minimum):
    """Mean island Moran's I from first principles: scipy components, then a
    double loop over the island's own cells (no sparse matrix, no restriction)."""
    labels, count = _scipy_labels(mask, connectivity)
    scores = []
    for island in range(1, count + 1):
        cells = np.argwhere(labels == island)
        if len(cells) < minimum or len(cells) < 2:
            continue
        member = np.array([values[tuple(c)] for c in cells])
        centred = member - member.mean()
        denominator = float((centred ** 2).sum())
        numerator, pairs = 0.0, 0
        for i, left in enumerate(cells):
            for j, right in enumerate(cells):
                if i == j:
                    continue
                gap = np.abs(left - right)
                adjacent = gap.max() == 1 if connectivity == "queen" else gap.sum() == 1
                if adjacent:
                    numerator += centred[i] * centred[j]
                    pairs += 1
        if pairs == 0 or denominator == 0.0:
            continue
        scores.append(len(cells) / pairs * numerator / denominator)
    return float(np.mean(scores)) if scores else float("nan")


class TestIslandsMoransI3D:
    @pytest.mark.parametrize("connectivity", ["queen", "rook"])
    def test_the_island_mean_matches_a_first_principles_loop(self, connectivity):
        values = _smooth_volume((7, 8, 10), seed=6, sigma=1.2)
        mask = values > 0.08
        expected = _islands_oracle(values, mask, connectivity, minimum=3)
        assert np.isfinite(expected), "the fixture must contain a scoreable island"
        assert islands_morans_i(
            values, mask, connectivity=connectivity, min_island_size=3
        ) == pytest.approx(expected, abs=1e-12)

    def test_no_island_is_not_applicable_in_3d_either(self):
        values = _smooth_volume((5, 5, 5), seed=7)
        assert np.isnan(islands_morans_i(values, np.zeros((5, 5, 5), dtype=bool)))


class TestMoransISummary3D:
    @pytest.mark.parametrize("connectivity", ["queen", "rook"])
    def test_the_summary_keeps_its_keys_and_scores_the_volume(self, connectivity):
        values = _smooth_volume((7, 8, 10), seed=6, sigma=1.2)
        mask = values > 0.08
        summary = morans_i_summary(values, mask, connectivity=connectivity)
        assert set(summary) == {
            "standard", "islands", "permutation_p", "num_permutations",
            "num_units", "connectivity",
        }
        assert summary["num_units"] == values.size == 560
        assert summary["connectivity"] == connectivity
        assert summary["standard"] == pytest.approx(
            _brute_force_morans_i(values, connectivity), abs=1e-12
        )
        assert summary["islands"] == pytest.approx(
            _islands_oracle(values, mask, connectivity, minimum=3), abs=1e-12
        )
        assert np.isnan(summary["permutation_p"])

    def test_the_permutation_p_value_matches_an_independently_built_matrix(self):
        """Same seed, same values, and a weight matrix built by a triple loop."""
        import scipy.sparse

        values = _smooth_volume((4, 5, 6), seed=8, sigma=1.0)
        shape = values.shape
        index = np.arange(values.size).reshape(shape)
        rows, cols = [], []
        for cell in np.ndindex(*shape):
            for step in np.ndindex(3, 3, 3):
                offset = tuple(s - 1 for s in step)
                other = tuple(c + o for c, o in zip(cell, offset))
                if any(offset) and all(0 <= x < n for x, n in zip(other, shape)):
                    rows.append(index[cell])
                    cols.append(index[other])
        weights = scipy.sparse.csr_matrix(
            (np.ones(len(rows)), (rows, cols)), shape=(values.size, values.size)
        )
        _, expected = morans_i_permutation_test(
            values.ravel(), weights, num_permutations=99, seed=5
        )
        summary = morans_i_summary(values, num_permutations=99, seed=5)
        assert summary["permutation_p"] == expected
        assert summary["num_permutations"] == 99
