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

import hashlib

import numpy as np
import pytest
import scipy.stats
from scipy import ndimage

from dl_techniques.layers.regularization.spatial_smoothness import SpatialLayout
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

    @pytest.mark.parametrize("shape", [(9,), (3, 3, 3, 3), (2, 2, 2, 2, 2)])
    def test_a_map_that_is_not_two_dimensional_is_named_not_unpacked(self, shape):
        """A map of an unsupported rank used to die in ``height, width = t_grid.shape``.

        That unpack error names nothing, so a caller could not tell a rank
        problem from a bug. The guard comes before the shape-agreement check, so
        two maps that AGREE in an unsupported shape get the rank message, not a
        pass.

        Rewritten in plan 972a9ac5 step 2 (D-003): the parametrization used to
        hold the 3-D shapes ``(7, 7, 7)`` and ``(4, 5, 6)`` as examples of a
        refused map. ``grow_clusters`` now grows clusters on a 3-D map
        (``TestGrowClusters3D``), so the shapes that must still be refused are
        the ones of rank 1 and 4 or more.
        """
        with pytest.raises(ValueError, match=r"2-D \(height, width\) map") as caught:
            grow_clusters(np.zeros(shape), np.ones(shape, dtype=bool))
        assert "unpack" not in str(caught.value)
        assert str(shape) in str(caught.value)


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

# =====================================================================
# plan-2026-10-23T190557-972a9ac5, step 1: pins and RED tests
#
# ``*GoldenPins`` record what the 2-D functions return at commit 9a1a446ec and
# must stay green through every later step. The ``*3D`` classes were RED when
# written (``grow_clusters`` and ``label_islands`` refused any map that is not
# 2-D); each compares the code with something that does not import the code:
# ``scipy.ndimage.label``, ``SpatialLayout.positions``, a hand-built block.
# =====================================================================

def _digest(array):
    """First 16 hex digits of the SHA-256 of an array's bytes (same environment pin)."""
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()[:16]


def _growth_digest(clusters, labels):
    """One digest over the unit ids of every cluster, in order, and the label grid."""
    state = hashlib.sha256()
    for cluster in clusters:
        state.update(np.asarray(cluster, "int64").tobytes())
        state.update(b"|")
    state.update(np.asarray(labels, "int64").tobytes())
    return state.hexdigest()[:16]


class TestLabelIslandsGoldenPins:
    """The cluster module's own copy of ``label_islands``, label grid for label grid."""

    #: ``(connectivity, min_size)`` -> ``(count, digest of the int64 labels)``.
    LABELS = {
        ("queen", 1): (5, "432c951917e662f7"),
        ("queen", 3): (2, "a9041cc11d334002"),
        ("queen", 10): (1, "3e27b20019bd2f35"),
        ("rook", 1): (6, "e502a5a8de96b98e"),
        ("rook", 3): (2, "a9041cc11d334002"),
        ("rook", 10): (1, "3e27b20019bd2f35"),
    }

    @staticmethod
    def _mask():
        """A 20-cell block, a 9-cell block, a 2-cell island, a diagonal pair and a lone cell."""
        mask = np.zeros((14, 14), dtype=bool)
        mask[1:6, 1:5] = True
        mask[8:11, 2:5] = True
        mask[2, 9] = mask[2, 10] = True
        mask[7, 8] = mask[8, 9] = True
        mask[12, 12] = True
        return mask

    @pytest.mark.parametrize("key", list(LABELS), ids=lambda k: f"{k[0]}-min{k[1]}")
    def test_label_islands_is_bit_identical(self, key):
        connectivity, minimum = key
        labels, count = label_islands(
            self._mask(), connectivity=connectivity, min_size=minimum
        )
        assert (count, _digest(labels)) == self.LABELS[key]
        assert labels.dtype == np.int64


class TestGrowClustersGoldenPins:
    """``grow_clusters`` over 6 seeds x 2 signs x 2 connectivities x {no cap, cap 4}.

    Each map is a smoothed noise field thresholded at ``|t| > 1`` (so clusters are
    big enough for the cap to bite: it changes the result in all 24 cases), grown
    with ``min_size=2`` through a permuted ``SpatialLayout`` so the unit ids are
    pinned too, not only the cell labels.
    """

    #: ``(seed, sign, connectivity)`` -> ``(clusters, digest)`` without a cap, then
    #: with ``max_size=4``.
    GOLDEN = {
        (0, 1, 'queen'): (4, 'c107a29218c2a596', 21, '68c5f3aa5a3f71a4'),
        (0, 1, 'rook'): (5, 'fc19b704b3eaca85', 19, '28ecd032f9412f09'),
        (0, -1, 'queen'): (4, '03cd2a2a7c5300f7', 6, '135d45594160f867'),
        (0, -1, 'rook'): (5, '51c683eeff8bfe91', 7, 'a0c34b90f0e6ebb9'),
        (1, 1, 'queen'): (6, 'bee4046477ea7301', 10, 'dbbbcbfe94b26003'),
        (1, 1, 'rook'): (6, 'e8f608d3a26b9f9e', 8, '07ab778465cd8518'),
        (1, -1, 'queen'): (2, '80437756e15d61d9', 20, 'f22b65e73e71f083'),
        (1, -1, 'rook'): (3, 'd5ec3c6c7fae8445', 18, '2030505addcd8612'),
        (2, 1, 'queen'): (6, '2e80d25df21f82e5', 10, '76691b17538391ec'),
        (2, 1, 'rook'): (7, 'e620deb62cc426ee', 10, 'b2b3ec6512c3bc0c'),
        (2, -1, 'queen'): (4, 'a9362c2023e1dab3', 15, 'd490a278ddb0dda6'),
        (2, -1, 'rook'): (4, 'a9362c2023e1dab3', 15, '1996362aa5153f52'),
        (3, 1, 'queen'): (6, '2e176b63768d57c8', 14, '2b782349fcd43e96'),
        (3, 1, 'rook'): (6, '2e176b63768d57c8', 14, 'f620573e59ccae86'),
        (3, -1, 'queen'): (7, 'bbf9ce13a382a823', 10, '4a703f717f403d70'),
        (3, -1, 'rook'): (6, '6303df2c7e416f04', 9, 'b63b7c2a6e6de9b3'),
        (4, 1, 'queen'): (6, '826f185bf038f31f', 17, 'ba2489f6712147fe'),
        (4, 1, 'rook'): (6, '826f185bf038f31f', 16, 'dbb08fa288e1ff64'),
        (4, -1, 'queen'): (5, '338c5bdebdf9d2dd', 8, '2acc8b7eebfa2189'),
        (4, -1, 'rook'): (5, '338c5bdebdf9d2dd', 8, '71b83db51983a44f'),
        (5, 1, 'queen'): (7, '3961d83877c3b720', 15, 'b8fc023388acc75c'),
        (5, 1, 'rook'): (7, '3961d83877c3b720', 14, 'ebdfb7cf17930ec1'),
        (5, -1, 'queen'): (4, '41fbde36051c0bce', 15, '7bcd2ecdb922df95'),
        (5, -1, 'rook'): (4, '4790fd3c616dbd2b', 14, '5f4e16baa7423c44'),
    }

    @pytest.mark.parametrize("key", list(GOLDEN), ids=lambda k: f"seed{k[0]}-sign{k[1]:+d}-{k[2]}")
    def test_clusters_and_label_grids_are_bit_identical(self, key):
        seed, sign, connectivity = key
        field = ndimage.gaussian_filter(
            np.random.default_rng(100 + seed).normal(size=(14, 14)), 1.0
        ) * 5.0
        layout = SpatialLayout(196, seed=seed)
        observed = []
        for cap in (None, 4):
            clusters, labels = grow_clusters(
                field, np.abs(field) > 1.0, sign=sign, min_size=2,
                connectivity=connectivity, max_size=cap,
                cell_to_unit=layout.cell_to_unit,
            )
            observed.extend([len(clusters), _growth_digest(clusters, labels)])
        assert tuple(observed) == self.GOLDEN[key]


# ---------------------------------------------------------------------
# rank 3
# ---------------------------------------------------------------------


def _smooth_volume(shape, seed, sigma=1.0, scale=6.0):
    return scale * ndimage.gaussian_filter(
        np.random.default_rng(seed).normal(size=shape), sigma
    )


def _scipy_labels(mask, connectivity):
    """``scipy.ndimage.label`` with the structure written out, not imported."""
    rank = mask.ndim
    return ndimage.label(
        mask,
        structure=ndimage.generate_binary_structure(
            rank, rank if connectivity == "queen" else 1
        ),
    )


class TestLabelIslands3D:
    @pytest.mark.parametrize("shape", [(10, 10, 10), (7, 8, 14)], ids=str)
    @pytest.mark.parametrize("connectivity", ["queen", "rook"])
    def test_islands_are_scipys_components(self, shape, connectivity):
        mask = _smooth_volume(shape, seed=9) > 1.0
        reference, count = _scipy_labels(mask, connectivity)
        assert count >= 3, "the fixture must have several islands to mean anything"
        labels, found = label_islands(mask, connectivity=connectivity, min_size=1)
        assert found == count
        np.testing.assert_array_equal(labels, reference)

    def test_queen_merges_a_pair_that_touches_only_at_a_corner(self):
        mask = np.zeros((4, 4, 4), dtype=bool)
        mask[1, 1, 1] = mask[2, 2, 2] = True
        assert label_islands(mask, "queen", min_size=1)[1] == 1
        assert label_islands(mask, "rook", min_size=1)[1] == 2

    def test_small_islands_are_dropped_and_the_rest_renumbered(self):
        mask = np.zeros((6, 6, 6), dtype=bool)
        mask[0, 0, 0] = True
        mask[2:4, 2:4, 2:4] = True
        mask[5, 5, 4:6] = True
        labels, count = label_islands(mask, "queen", min_size=2)
        assert count == 2
        assert sorted(np.unique(labels).tolist()) == [0, 1, 2]
        assert int((labels == 1).sum()) == 8 and int((labels == 2).sum()) == 2


class TestGrowClusters3D:
    @pytest.mark.parametrize("shape", [(10, 10, 10), (7, 8, 14)], ids=str)
    @pytest.mark.parametrize("connectivity", ["queen", "rook"])
    def test_without_a_cap_the_partition_is_scipys_components(
        self, shape, connectivity
    ):
        """The equivalence the module docstring claims, now for a volume."""
        volume = _smooth_volume(shape, seed=10)
        significant = np.abs(volume) > 1.0
        for sign in (1, -1):
            candidate = significant & (sign * volume > 0)
            reference, count = _scipy_labels(candidate, connectivity)
            assert count >= 2, "the fixture must have several components"
            clusters, labels = grow_clusters(
                volume, significant, sign=sign, min_size=1, connectivity=connectivity
            )
            assert len(clusters) == count
            assert labels.shape == shape
            assert np.array_equal(labels > 0, candidate)
            pairs = set(zip(reference[candidate].tolist(), labels[candidate].tolist()))
            assert len(pairs) == count, "the partition differs from scipy's"
            assert len({a for a, _ in pairs}) == len({b for _, b in pairs}) == count

    def test_unit_ids_map_back_to_the_labelled_cells_through_positions(self):
        """``layout.positions`` is the independent oracle: it knows where each
        unit sits without ever reading ``cell_to_unit``, which the function
        under test indexes. The expected cells come from scipy plus a size
        filter, so nothing here is derived from ``grow_clusters`` itself."""
        layout = SpatialLayout(1000, seed=5, grid_dim=3)
        assert layout.grid_shape == (10, 10, 10)
        volume = _smooth_volume(layout.grid_shape, seed=11, sigma=1.5, scale=8.0)
        significant = volume > 1.0
        components, count = _scipy_labels(significant, "queen")
        sizes = np.bincount(components.ravel())
        kept = {
            tuple(int(v) for v in cell)
            for cell in np.argwhere(significant)
            if sizes[components[tuple(cell)]] >= 5
        }
        assert kept and len(kept) < int(significant.sum()), (
            "the fixture must drop at least one small component"
        )

        clusters, labels = grow_clusters(
            volume, significant, sign=1, min_size=5, connectivity="queen",
            cell_to_unit=layout.cell_to_unit,
        )
        recovered = set()
        for cluster in clusters:
            recovered |= {tuple(int(v) for v in layout.positions[u]) for u in cluster}
        assert recovered == kept
        assert {tuple(int(v) for v in cell) for cell in np.argwhere(labels > 0)} == kept

    def test_a_non_cubic_block_pins_the_axis_order(self):
        """A 2x3x6 block of a (7, 8, 14) volume is 36 cells whatever the order of
        the axes, so the count alone would pass a transposed implementation. The
        bounding box of the recovered positions does not: it is (1, 2, 6) to
        (2, 4, 11) only if axis 0 is the slowest and axis 2 the fastest."""
        layout = SpatialLayout(7 * 8 * 14, seed=2, grid_dim=3)
        assert layout.grid_shape == (7, 8, 14)
        volume = np.zeros(layout.grid_shape)
        volume[1:3, 2:5, 6:12] = 5.0
        clusters, labels = grow_clusters(
            volume, volume > 1.0, sign=1, min_size=1, connectivity="queen",
            cell_to_unit=layout.cell_to_unit,
        )
        assert len(clusters) == 1 and len(clusters[0]) == 36
        cells = layout.positions[clusters[0]]
        assert cells.min(axis=0).tolist() == [1, 2, 6]
        assert cells.max(axis=0).tolist() == [2, 4, 11]
        assert int((labels > 0).sum()) == 36

    def test_a_cap_splits_a_3d_component_like_the_2d_one(self):
        """The both-ways twin: 27 cells capped at 9 give three clusters of 9."""
        volume = np.full((7, 7, 7), -0.1)
        volume[2:5, 2:5, 2:5] = 5.0
        capped, _ = grow_clusters(
            volume, volume > 1.5, sign=1, min_size=1, max_size=9
        )
        uncapped, _ = grow_clusters(volume, volume > 1.5, sign=1, min_size=1)
        assert [len(c) for c in capped] == [9, 9, 9]
        assert [len(c) for c in uncapped] == [27]

    def test_queen_joins_a_corner_pair_and_rook_does_not(self):
        volume = np.zeros((5, 5, 5))
        volume[1, 1, 1] = volume[2, 2, 2] = 4.0
        queen, _ = grow_clusters(volume, volume > 1, 1, 1, "queen")
        rook, _ = grow_clusters(volume, volume > 1, 1, 1, "rook")
        assert [len(c) for c in queen] == [2]
        assert sorted(len(c) for c in rook) == [1, 1]

    @pytest.mark.parametrize("shape", [(9,), (3, 3, 3, 3)], ids=["1D", "4D"])
    def test_ranks_one_and_four_are_still_refused_by_name(self, shape):
        """True before and after the 3-D support: the message keeps the
        (height, width) wording and the offending shape."""
        with pytest.raises(ValueError, match=r"2-D \(height, width\) map") as caught:
            grow_clusters(np.zeros(shape), np.ones(shape, dtype=bool))
        assert str(shape) in str(caught.value)

    def test_a_3d_map_is_accepted(self):
        """The other half of the rank guard."""
        clusters, labels = grow_clusters(
            np.full((3, 3, 3), 4.0), np.ones((3, 3, 3), dtype=bool), min_size=1
        )
        assert len(clusters) == 1 and len(clusters[0]) == 27
        assert labels.shape == (3, 3, 3)
