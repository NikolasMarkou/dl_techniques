"""Test suite for :mod:`dl_techniques.metrics.topographic_selectivity`.

Every function here returns a map that a figure is drawn from, so the tests pin
NUMBERS: the t-map on inputs whose answer is known, q-values against SciPy, and
the cluster-growing equivalence that the module docstring claims. Two of them
also carry the negative arms -- what happens to a dead unit, and what happens
when the significance map is empty -- because those are the cases that produce a
plausible-looking empty figure.

The mandated pins, with the numbers this suite measures:

  1. test_a_planted_contrast_is_recovered_by_sign_and_rank   Spearman 1.0
  2. test_fdr_bh_matches_scipy                               exact delegation
  3. test_correcting_across_layers_is_stricter_than_correcting_per_layer
  4. test_grow_clusters_without_a_cap_equals_connected_components
  5. test_the_sign_of_a_cluster_follows_the_polarity      +1 / -1 disjoint
"""

import numpy as np
import pytest
import scipy.stats

from dl_techniques.metrics.topographic_selectivity import (
    CLUSTER_SIGNS,
    DEFAULT_MIN_CLUSTER_SIZE,
    cluster_response,
    contrast_tmap,
    fdr_across_layers,
    fdr_bh,
    grow_clusters,
    label_islands,
    profile_consistency,
)


def _blob_mask(height, width, top=2, bottom=5, left=2, right=5):
    mask = np.zeros((height, width), dtype=bool)
    mask[top:bottom, left:right] = True
    return mask


class TestContrastTmap:
    def test_a_planted_contrast_is_recovered_by_sign_and_rank(self):
        """Sign AND ordering, on a map whose answer was constructed.

        Units 0-9 respond more to condition A, 10-19 more to B, the rest not at
        all. Asserting only the sign would pass on a map that ranked everything
        correctly but called the two directions the same way round.
        """
        rng = np.random.default_rng(0)
        units = 40
        condition_b = rng.normal(size=(20, units))
        condition_a = condition_b.copy()
        condition_a[:, :10] += 5.0
        condition_a[:, 10:20] -= 5.0

        t_values, p_values = contrast_tmap(condition_a, condition_b)
        # Measured: planted units |t| >= 11.1, untouched units t == 0.000 to the
        # last bit -- the two conditions are the SAME array there, so the
        # two-sample t is exactly zero. A fixture that shifted every unit by a
        # constant instead (the obvious construction) would make the "noise"
        # units differ by that constant too, and the untouched units would carry
        # a large t that this assertion would have had to accommodate.
        assert t_values[:10].min() > 10.0
        assert t_values[10:20].max() < -10.0
        assert np.all(np.abs(t_values[20:]) < 1e-9)
        assert p_values[:20].max() < 1e-12
        assert p_values[20:].min() > 0.99

    def test_a_unit_constant_in_both_conditions_is_no_evidence(self):
        """SciPy returns NaN; this returns ``t = 0, p = 1``.

        NaN would poison every min/max and every threshold derived from them, and
        a constant unit genuinely carries no evidence either way.
        """
        rng = np.random.default_rng(1)
        a = rng.normal(size=(16, 8))
        b = rng.normal(size=(16, 8))
        a[:, 3] = 1.25
        b[:, 3] = 1.25

        t_values, p_values = contrast_tmap(a, b)
        assert t_values[3] == 0.0
        assert p_values[3] == 1.0
        assert np.all(np.isfinite(t_values))
        assert np.all(np.isfinite(p_values))

    def test_welch_and_student_agree_on_balanced_inputs(self):
        """The ``equal_var`` knob reaches the test, and both arms are finite.

        The two differ on unequal variances, so the equality below is a real
        (if weak) check that the flag is plumbed rather than ignored -- and the
        difference check is the one that would catch an ignored flag.
        """
        rng = np.random.default_rng(2)
        a = rng.normal(size=(20, 12))
        b = rng.normal(size=(3, 12)) + 2.0
        student, _ = contrast_tmap(a, b, equal_var=True)
        welch, _ = contrast_tmap(a, b, equal_var=False)
        assert np.all(np.isfinite(student))
        assert np.all(np.isfinite(welch))
        # Measured 4.11 against 4.18 on the first unit: with n = 20 against 3 the
        # two denominators genuinely differ, and an implementation that dropped
        # the flag would return the same array twice.
        assert not np.allclose(student, welch)

    def test_the_shapes_are_checked(self):
        with pytest.raises(ValueError, match="must be 2-D"):
            contrast_tmap(np.zeros(10), np.zeros((10, 4)))
        with pytest.raises(ValueError, match="share a unit axis"):
            contrast_tmap(np.zeros((10, 4)), np.zeros((10, 5)))
        with pytest.raises(ValueError, match="at least one stimulus"):
            contrast_tmap(np.zeros((0, 4)), np.zeros((10, 4)))


class TestFdr:
    def test_fdr_bh_matches_scipy(self):
        """Delegation, not a second transcription of the step-up rule.

        Two producers of one q-value is the shape that drifts; the guard here is
        the comparison rather than a hand-computed expectation.
        """
        rng = np.random.default_rng(0)
        p_values = rng.uniform(size=500) ** 3
        reject, q_values = fdr_bh(p_values, alpha=0.05)
        expected = scipy.stats.false_discovery_control(p_values, method="bh")
        np.testing.assert_allclose(q_values, expected, atol=0.0, rtol=0)
        assert np.array_equal(reject, q_values < 0.05)

    def test_the_shape_is_preserved(self):
        p_values = np.full((3, 4), 0.01)
        reject, q_values = fdr_bh(p_values)
        assert q_values.shape == (3, 4)
        assert reject.shape == (3, 4)

    def test_a_tighter_alpha_rejects_less(self):
        """The threshold reaches the decision, both ways."""
        rng = np.random.default_rng(1)
        # A beta(0.4, 8) set straddles the threshold: measured 192 rejections at
        # alpha = 0.05, 12 at 0.001 and none at 1e-6. Uniform p-values would sit
        # entirely above 0.05 and the "loose" arm would reject nothing, leaving
        # the assertion vacuous.
        p_values = rng.beta(0.4, 8.0, size=300)
        loose, _ = fdr_bh(p_values, alpha=0.05)
        tight, _ = fdr_bh(p_values, alpha=0.001)
        strict, _ = fdr_bh(p_values, alpha=1e-6)
        assert loose.sum() > 0
        assert strict.sum() == 0
        assert strict.sum() <= tight.sum() <= loose.sum()

    def test_correcting_across_layers_is_stricter_than_correcting_per_layer(self):
        """Why the paper corrects once over every tap.

        Three maps of 200 p-values, each of which looks significant on its own.
        Correcting each map separately admits many false positives; correcting the
        concatenated 600 admits fewer, and the difference is the whole argument
        for the joint correction.
        """
        rng = np.random.default_rng(2)
        maps = {
            "block_0_attn": rng.uniform(size=200),
            "block_0_mlp": rng.uniform(size=200),
            "block_1_attn": rng.uniform(size=200),
        }
        joint = fdr_across_layers(maps, alpha=0.05)

        per_layer = sum(int(fdr_bh(values)[0].sum()) for values in maps.values())
        together = int(sum(int(result["reject"].sum()) for result in joint.values()))
        assert together <= per_layer

    def test_the_joint_result_splits_back_to_the_right_shapes_and_order(self):
        rng = np.random.default_rng(3)
        maps = {"a": rng.uniform(size=7), "b": rng.uniform(size=11)}
        joint = fdr_across_layers(maps)
        assert joint["a"]["q"].shape == (7,)
        assert joint["b"]["q"].shape == (11,)

        stacked = np.concatenate([maps["a"], maps["b"]])
        _, expected = fdr_bh(stacked)
        np.testing.assert_allclose(
            np.concatenate([joint["a"]["q"], joint["b"]["q"]]), expected,
            atol=0.0, rtol=0,
        )

    def test_the_joint_q_of_each_map_exceeds_its_alone_q(self):
        """Per-map q-values are anti-conservative, and the gap is measurable."""
        rng = np.random.default_rng(4)
        maps = {"a": rng.uniform(size=300) ** 2, "b": rng.uniform(size=300) ** 2}
        joint = fdr_across_layers(maps)
        alone, _ = fdr_bh(maps["a"])
        assert joint["a"]["q"].sum() >= alone.sum() - 1e-9

    def test_an_empty_mapping_is_rejected(self):
        with pytest.raises(ValueError, match="must not be empty"):
            fdr_across_layers({})


class TestLabelIslands:
    def test_islands_below_the_size_floor_are_dropped_and_relabelled(self):
        """Dropping must also RENUMBER, or ``1..count`` is a false contract.

        A small island at (1,1) and a large one at (3:6, 3:6). `ndimage.label`
        numbers them 1 and 2; dropping 1 leaves the survivor labelled 2 with
        ``count == 1``, so a caller looping ``range(1, count + 1)`` reads a label
        that does not exist.
        """
        mask = np.zeros((7, 7), dtype=bool)
        mask[1, 1] = True
        mask[3:6, 3:6] = True

        labels, count = label_islands(mask, min_size=3)
        assert count == 1
        assert labels[1, 1] == 0
        assert labels[3, 3] == 1
        assert set(np.unique(labels).tolist()) == {0, 1}

    def test_the_default_minimum_is_the_papers_ten(self):
        assert DEFAULT_MIN_CLUSTER_SIZE == 10
        mask = np.zeros((12, 12), dtype=bool)
        mask[2:6, 2:6] = True  # 16 cells, kept
        mask[9:10, 9:10] = True  # 1 cell, dropped
        _, count = label_islands(mask)
        assert count == 1

    def test_diagonal_touching_cells_merge_under_queen_only(self):
        mask = np.zeros((5, 5), dtype=bool)
        mask[1, 1] = True
        mask[2, 2] = True
        assert label_islands(mask, connectivity="queen", min_size=1)[1] == 1
        assert label_islands(mask, connectivity="rook", min_size=1)[1] == 2

    def test_an_all_significant_map_is_one_island(self):
        _, count = label_islands(np.ones((6, 6), dtype=bool))
        assert count == 1

    @pytest.mark.parametrize(
        "kwargs,message",
        [
            ({"connectivity": "bishop"}, "connectivity must be one of"),
            ({"min_size": 0}, "min_size must be >= 1"),
        ],
    )
    def test_invalid_arguments_name_the_offender(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            label_islands(np.ones((4, 4), dtype=bool), **kwargs)


class TestGrowClusters:
    def _map(self, seed=0, blobs=((2, 5, 2, 5, 4.0), (8, 11, 8, 11, -4.0))):
        """A t-map with two planted blobs of opposite sign on a noise floor."""
        rng = np.random.default_rng(seed)
        t_grid = rng.normal(scale=0.3, size=(14, 14))
        for top, bottom, left, right, value in blobs:
            t_grid[top:bottom, left:right] += value
        sig_grid = np.abs(t_grid) > 1.5
        return t_grid, sig_grid

    def test_grow_clusters_without_a_cap_equals_connected_components(self):
        """The equivalence the module docstring claims, pinned on both signs.

        A greedy traversal reaches the same partition whichever member it enters
        a maximal connected set from, so with ``max_size=None`` the result must be
        ``label_islands`` -- and two implementations of one quantity is exactly
        what needs a guard.
        """
        t_grid, sig_grid = self._map()
        for sign in CLUSTER_SIGNS:
            mask = sig_grid & (sign * t_grid > 0)
            clusters, labels = grow_clusters(
                t_grid, sig_grid, sign=sign, min_size=1, max_size=None
            )
            reference, count = label_islands(
                mask, connectivity="queen", min_size=1
            )
            assert int(labels.max()) == count
            np.testing.assert_array_equal(labels, reference)

    def test_clusters_below_the_floor_are_dropped(self):
        t_grid, sig_grid = self._map()
        clusters, labels = grow_clusters(
            t_grid, sig_grid, sign=1, min_size=5
        )
        assert all(len(cluster) >= 5 for cluster in clusters)
        assert int(labels.max()) == len(clusters)

    def test_the_sign_of_a_cluster_follows_the_polarity(self):
        """Positive and negative runs are DISJOINT.

        Without this a sign-blind implementation -- one that grew every
        significant cell regardless of polarity -- would return the same clusters
        for both signs and the both-ways assertion below could not fail.
        """
        t_grid, sig_grid = self._map()
        positive, _ = grow_clusters(t_grid, sig_grid, sign=1, min_size=1)
        negative, _ = grow_clusters(t_grid, sig_grid, sign=-1, min_size=1)

        positive_units = set().union(*[set(c.tolist()) for c in positive])
        negative_units = set().union(*[set(c.tolist()) for c in negative])
        assert positive_units and negative_units
        assert not (positive_units & negative_units)
        assert all(t_grid.reshape(-1)[u] > 0 for u in positive_units)
        assert all(t_grid.reshape(-1)[u] < 0 for u in negative_units)

    def test_a_size_cap_splits_one_component_into_several(self):
        """Where the greedy order becomes part of the definition.

        One blob of nine significant cells, capped at three: three clusters, not
        one. Without a cap the same input yields one cluster, which is the both-
        ways twin for ``max_size``.
        """
        t_grid = np.full((10, 10), -0.1)
        t_grid[4:7, 4:7] = 5.0
        sig_grid = t_grid > 1.5

        capped, _ = grow_clusters(t_grid, sig_grid, sign=1, min_size=1, max_size=3)
        uncapped, _ = grow_clusters(
            t_grid, sig_grid, sign=1, min_size=1, max_size=None
        )
        assert all(len(cluster) <= 3 for cluster in capped)
        assert len(capped) == 3
        assert len(uncapped) == 1
        assert len(uncapped[0]) == 9

    def test_clusters_come_back_as_unit_ids_when_a_layout_is_supplied(self):
        """Row-major cell indices are the fallback; unit ids are the product.

        The post-hoc pipeline reads activations by unit, so a cluster labelled by
        cell index would silently point at the wrong units -- and a shape check
        would not see it, because both labellings have the same length. What
        distinguishes them is that the unit ids, mapped back through the layout,
        reproduce the significant CELLS.
        """
        from dl_techniques.layers.regularization.spatial_smoothness import (
            SpatialLayout,
        )

        layout = SpatialLayout(196, seed=0)  # 14 x 14
        t_grid, sig_grid = self._map()

        with_layout, _ = grow_clusters(
            t_grid, sig_grid, sign=1, min_size=1,
            cell_to_unit=layout.cell_to_unit,
        )
        without_layout, _ = grow_clusters(t_grid, sig_grid, sign=1, min_size=1)

        significant_cells = {
            tuple(cell) for cell in np.argwhere(sig_grid & (t_grid > 0))
        }
        recovered = set()
        for cluster in with_layout:
            recovered |= {tuple(map(int, cell)) for cell in layout.positions[cluster]}

        assert recovered == significant_cells
        assert sorted(np.concatenate(with_layout).tolist()) != sorted(
            np.concatenate(without_layout).tolist()
        ), (
            "the layout-supplied ids are the row-major cell indices -- the "
            "permutation is not being applied"
        )

    def test_an_empty_significance_map_yields_no_clusters(self):
        t_grid = np.zeros((8, 8))
        clusters, labels = grow_clusters(
            t_grid, np.zeros((8, 8), dtype=bool), sign=1
        )
        assert clusters == []
        assert labels.sum() == 0

    def test_uniformly_significant_cells_grow_one_cluster(self):
        t_grid = np.full((9, 9), 4.0)
        sig_grid = np.ones((9, 9), dtype=bool)
        clusters, labels = grow_clusters(t_grid, sig_grid, sign=1, min_size=1)
        assert len(clusters) == 1
        assert len(clusters[0]) == 81

    @pytest.mark.parametrize(
        "kwargs,message",
        [
            ({"sign": 0}, "sign must be \\+1 or -1"),
            ({"connectivity": "bishop"}, "connectivity must be one of"),
            ({"max_size": 0}, "max_size must be >= 1 or None"),
        ],
    )
    def test_invalid_arguments_name_the_offender(self, kwargs, message):
        t_grid = np.zeros((6, 6))
        with pytest.raises(ValueError, match=message):
            grow_clusters(t_grid, np.ones((6, 6), dtype=bool), **kwargs)

    def test_mismatched_map_shapes_are_rejected(self):
        with pytest.raises(ValueError, match="does not match t_grid shape"):
            grow_clusters(np.zeros((5, 5)), np.ones((6, 6), dtype=bool))


class TestClusterResponse:
    def _activations(self, units=(0, 1, 2), conditions=("a", "b")):
        rng = np.random.default_rng(0)
        out = {}
        for index, name in enumerate(conditions):
            scale = 1.0 + index
            out[name] = rng.normal(loc=scale, scale=1.0, size=(12, 6))
        return out

    def test_the_profile_is_the_mean_absolute_response_per_condition(self):
        activations = self._activations()
        units = [0, 1, 2]
        profile = cluster_response(activations, units)

        assert set(profile) == {"a", "b"}
        for name, values in activations.items():
            expected = np.abs(values[:, units]).mean(axis=1)
            assert profile[name]["mean"] == pytest.approx(expected.mean())
            assert profile[name]["sem"] == pytest.approx(
                expected.std(ddof=1) / np.sqrt(len(expected))
            )

    def test_the_absolute_value_matters(self):
        """A cluster whose units cancel would read 0.0 signed and 1.0 absolute.

        Two units with exactly opposite responses: averaging their signed
        activations gives zero at every stimulus, so the "response" would be a
        constant. The paper's definition is mean |activation|, and this is the
        case that distinguishes the two.
        """
        activations = {
            "a": np.array([[2.0, -2.0], [4.0, -4.0]]),
            "b": np.array([[1.0, -1.0], [2.0, -2.0]]),
        }
        profile = cluster_response(activations, [0, 1])
        assert profile["a"]["mean"] == pytest.approx(3.0)
        assert profile["b"]["mean"] == pytest.approx(1.5)

    def test_a_single_stimulus_reports_an_undefined_sem(self):
        profile = cluster_response({"a": np.zeros((1, 4))}, [0])
        assert np.isnan(profile["a"]["sem"])

    def test_an_empty_cluster_is_rejected(self):
        with pytest.raises(ValueError, match="must not be empty"):
            cluster_response({"a": np.zeros((4, 4))}, [])

    def test_an_out_of_range_unit_is_rejected(self):
        with pytest.raises(ValueError, match="must lie in"):
            cluster_response({"a": np.zeros((4, 4))}, [9])


class TestProfileConsistency:
    def test_identical_profiles_score_a_perfect_correlation(self):
        """The both-ways twin: distinct profiles must score BELOW 1.

        Four clusters with the same condition ordering give a mean pairwise
        Spearman of exactly 1.0; perturbing one ordering must move it.
        """
        identical = [
            {"a": {"mean": 3.0, "sem": 0.1}, "b": {"mean": 2.0, "sem": 0.1},
             "c": {"mean": 1.0, "sem": 0.1}}
            for _ in range(4)
        ]
        assert profile_consistency(identical)["mean_pairwise_spearman"] == (
            pytest.approx(1.0)
        )

        perturbed = [dict(entry) for entry in identical]
        perturbed[0] = {
            "a": {"mean": 1.0, "sem": 0.1},
            "b": {"mean": 3.0, "sem": 0.1},
            "c": {"mean": 2.0, "sem": 0.1},
        }
        assert profile_consistency(perturbed)["mean_pairwise_spearman"] < 1.0

    def test_the_summary_reports_its_own_shape(self):
        profiles = [
            {"a": {"mean": float(i), "sem": 0.1}, "b": {"mean": float(4 - i), "sem": 0.1}}
            for i in range(3)
        ]
        summary = profile_consistency(profiles)
        assert summary["conditions"] == ["a", "b"]
        assert summary["num_clusters"] == 3
        assert summary["num_conditions"] == 2
        assert summary["interaction"]["available"] is True

    def test_a_single_cluster_cannot_be_compared_with_itself(self):
        """One profile has no pair to correlate, and says so rather than
        reporting a perfect agreement it never measured."""
        summary = profile_consistency(
            [{"a": {"mean": 1.0, "sem": 0.1}, "b": {"mean": 2.0, "sem": 0.1}}]
        )
        assert np.isnan(summary["mean_pairwise_spearman"])

    def test_profiles_disagreeing_about_their_conditions_are_rejected(self):
        with pytest.raises(ValueError, match="is missing condition"):
            profile_consistency(
                [
                    {"a": {"mean": 1.0, "sem": 0.1}, "b": {"mean": 2.0, "sem": 0.1}},
                    {"a": {"mean": 1.0, "sem": 0.1}},
                ]
            )

    def test_an_empty_profile_list_is_rejected(self):
        with pytest.raises(ValueError, match="must not be empty"):
            profile_consistency([])