"""RED proof for the transcription oracle in ``spatial_smoothness_oracle``.

A guard that cannot fail is the most likely outcome of writing a new test, and a
shared instrument is exactly the kind of code that gets trusted without being
tested. This module proves the oracle can reject, by mutating the *paper's
formula* in each of the ways an implementation could plausibly get wrong and
checking the oracle notices.

The mutations here are all inside the oracle, deliberately: the claim under test
is "this oracle discriminates", not "the implementation is right". The
implementation's own correctness is carried by
``test_the_loss_matches_the_paper_transcription_term_for_term``, which fails if
the two ever stop agreeing.

References:
    - Rathi et al., 2025. TopoLM, Eq. 1. (https://arxiv.org/abs/2410.11516)
"""

import types

import numpy as np
import pytest

from dl_techniques.layers.regularization.spatial_smoothness import (
    SpatialLayout,
    build_patch_tables,
)

from .spatial_smoothness_oracle import (
    reference_d_unit,
    reference_d_unit_3d,
    reference_patch_units_3d,
    reference_smoothness_loss,
    smooth_field_activations,
    smooth_field_activations_3d,
)

NUM_UNITS = 784
RADIUS = 2


@pytest.fixture(scope="module")
def fixture():
    layout = SpatialLayout(NUM_UNITS, seed=0)
    tables = build_patch_tables(layout.cell_to_unit, RADIUS, "linf")
    activations = smooth_field_activations(layout, num_samples=48, seed=0)
    return layout, tables, activations


class TestTheOracleCanDiscriminate:
    def test_the_oracle_is_not_vacuous_on_its_own_fixture(self, fixture):
        """Anti-vacuity first: the oracle must return a real, non-trivial number.

        An oracle that returns a constant satisfies every equality downstream
        while measuring nothing.
        """
        _, tables, activations = fixture
        value = reference_smoothness_loss(
            activations, tables.patch_table[0], tables.d_unit
        )
        assert 0.0 < value < 1.0

    def test_the_oracle_separates_a_smooth_field_from_noise(self, fixture):
        _, tables, activations = fixture
        rng = np.random.default_rng(0)
        smooth = reference_smoothness_loss(
            activations, tables.patch_table[0], tables.d_unit
        )
        noisy = reference_smoothness_loss(
            rng.normal(size=activations.shape),
            tables.patch_table[0],
            tables.d_unit,
        )
        assert smooth < noisy - 0.05, (smooth, noisy)

    def test_a_reversed_correlation_sign_is_rejected(self, fixture):
        """Sign of the correlation.

        Pearson is odd in either argument, so negating the measured
        correlations must land exactly on ``1 - SL`` about 0.5. If the oracle were
        insensitive to the correlation's sign, the mutation below would pass.
        """
        _, tables, activations = fixture
        correct = reference_smoothness_loss(
            activations, tables.patch_table[0], tables.d_unit
        )
        mutated = reference_smoothness_loss(
            activations,
            tables.patch_table[0],
            -tables.d_unit,
        )
        assert correct < 0.5
        assert mutated > 0.5
        assert correct + mutated == pytest.approx(1.0, abs=1e-9)

    def test_a_swapped_patch_is_rejected(self, fixture):
        """The patch must be the one asked for.

        A neighbourhood that silently read a fixed centre -- the classic
        off-by-one in a gather -- still returns a plausible number, so the
        discriminator has to be that DIFFERENT patches give DIFFERENT answers.

        The field is periodic over the grid, so a minority of sampled windows are
        congruent by construction; the bound is 80% distinct rather than "all
        distinct", and one guaranteed-distinct pair (a centre against its
        diagonal corner) is asserted outright.
        """
        _, tables, activations = fixture
        rows = list(range(0, tables.num_centers, 21))
        values = [
            reference_smoothness_loss(
                activations, tables.patch_table[row], tables.d_unit
            )
            for row in rows
        ]
        distinct = len(set(np.round(values, 6)))
        assert distinct >= 0.8 * len(values), (
            f"only {distinct} of {len(values)} neighbourhoods gave distinct "
            f"answers -- the oracle may be reading a fixed window"
        )

        first = reference_smoothness_loss(
            activations, tables.patch_table[0], tables.d_unit
        )
        far = reference_smoothness_loss(
            activations, tables.patch_table[tables.num_centers - 1], tables.d_unit
        )
        assert first != pytest.approx(far, abs=1e-6)

    def test_a_missing_normalisation_is_rejected(self, fixture):
        """The `0.5` factor and the centring are load-bearing.

        Dropping the factor doubles the loss; dropping the centring of ``d``
        changes it in a direction the sign test cannot catch. Both must move the
        number by far more than the tolerance the implementation agrees to.
        """
        _, tables, activations = fixture
        correct = reference_smoothness_loss(
            activations, tables.patch_table[0], tables.d_unit
        )

        uncentred = 1.0 / (np.arange(tables.num_pairs, dtype=np.float64) % 11 + 1.0)
        mutated = reference_smoothness_loss(
            activations, tables.patch_table[0], uncentred.astype(np.float32)
        )
        assert abs(correct - mutated) > 1e-3

    def test_an_unnormalized_prior_is_rejected(self, fixture):
        """Scale-invariance of the correlation must not hide a scale bug.

        ``corr`` is invariant to scaling ``d``, so multiplying the prior by a
        constant cannot be detected by any loss-value comparison -- which is
        exactly why the implementation normalises it once in NumPy rather than
        recomputing the norm per step. The oracle agrees on the VALUE of the
        normalized prior and disagrees on an unnormalized one.
        """
        _, tables, activations = fixture
        assert reference_d_unit(tables.patch_units) == pytest.approx(
            tables.d_unit, abs=1e-6
        )
        # A constant multiple of a unit vector is a different array, and the
        # equality above is what says the oracle normalises the same way.
        assert not np.allclose(reference_d_unit(tables.patch_units) * 3.0, tables.d_unit)


# ---------------------------------------------------------------------
# 3D transcriptions
#
# These tests prove the 3D helpers can reject, and they use NOTHING from the
# implementation: the layout is a two-line stand-in (a permutation and its
# inverse reshaped to the grid), the patches and priors come from the oracle
# itself, so they stay green and meaningful whether or not a 3D layer exists. The
# agreement of the implementation with these helpers is asserted separately, in
# ``test_spatial_smoothness_3d.py``.
# ---------------------------------------------------------------------

GRID_3D = (7, 8, 9)
NUM_UNITS_3D = 7 * 8 * 9
RADIUS_3D = 2
PATCH_3D = (2 * RADIUS_3D + 1) ** 3


def _stand_in_layout_3d(grid_shape=GRID_3D, seed=0):
    """The two attributes the 3D field generator reads, built by hand."""
    num_units = int(np.prod(grid_shape))
    perm = np.random.default_rng(seed).permutation(num_units).astype("int32")
    return types.SimpleNamespace(
        grid_shape=grid_shape,
        perm=perm,
        cell_to_unit=np.argsort(perm).reshape(grid_shape).astype("int32"),
    )


def _axis_blind_d_unit(side, dropped_axis):
    """A prior that forgets one of the three axes: the 2D-in-3D slip.

    Written separately from :func:`reference_d_unit_3d` so the two cannot share a
    bug; l-infinity over the two remaining offsets.
    """
    cells = [(a, b, c) for a in range(side) for b in range(side) for c in range(side)]
    pairs = []
    for i in range(len(cells)):
        for j in range(i + 1, len(cells)):
            kept = [
                abs(cells[i][k] - cells[j][k]) for k in range(3) if k != dropped_axis
            ]
            pairs.append(1.0 / (max(kept) + 1.0))
    pairs = np.array(pairs)
    pairs = pairs - pairs.mean()
    return (pairs / np.linalg.norm(pairs)).astype("float32")


@pytest.fixture(scope="module")
def fixture_3d():
    layout = _stand_in_layout_3d()
    patch = reference_patch_units_3d(layout.cell_to_unit, (3, 3, 4), RADIUS_3D)
    d_unit = reference_d_unit_3d(PATCH_3D, "linf")
    activations = smooth_field_activations_3d(layout, num_samples=48, seed=0)
    return layout, patch, d_unit, activations


class TestTheThreeDimensionalOracleCanDiscriminate:
    """Fixture: 7x8x9 stand-in grid, radius 2 (125-unit cubes), 48 samples.

    Measured on it: smooth field 0.204 at centre (2, 3, 4), noise 0.498, the
    negated prior 0.796.
    """

    def test_the_oracle_is_not_vacuous_on_its_own_fixture(self, fixture_3d):
        _, patch, d_unit, activations = fixture_3d
        assert patch.shape == (PATCH_3D,)
        value = reference_smoothness_loss(activations, patch, d_unit)
        assert 0.0 < value < 0.30, value

    def test_the_oracle_separates_a_smooth_field_from_noise(self, fixture_3d):
        _, patch, d_unit, activations = fixture_3d
        noise = np.random.default_rng(0).normal(size=activations.shape)
        smooth = reference_smoothness_loss(activations, patch, d_unit)
        noisy = reference_smoothness_loss(noise, patch, d_unit)
        assert smooth < noisy - 0.15, (smooth, noisy)
        assert abs(noisy - 0.5) < 0.05, noisy

    def test_a_reversed_correlation_sign_is_rejected(self, fixture_3d):
        _, patch, d_unit, activations = fixture_3d
        correct = reference_smoothness_loss(activations, patch, d_unit)
        mutated = reference_smoothness_loss(activations, patch, -d_unit)
        assert correct < 0.5
        assert mutated > 0.5
        assert correct + mutated == pytest.approx(1.0, abs=1e-9)

    def test_a_swapped_patch_is_rejected(self, fixture_3d):
        """Different cubes must give different answers.

        Admissible centres run over 2..4 x 2..5 x 2..6 (60 of them); the field
        is periodic, so a few windows can coincide, hence 80% distinct plus one
        guaranteed-distinct pair of opposite corners.
        """
        layout, _, d_unit, activations = fixture_3d
        centres = [
            (c0, c1, c2)
            for c0 in range(2, 5) for c1 in range(2, 6) for c2 in range(2, 7)
        ][::7]
        values = [
            reference_smoothness_loss(
                activations,
                reference_patch_units_3d(layout.cell_to_unit, centre, RADIUS_3D),
                d_unit,
            )
            for centre in centres
        ]
        distinct = len(set(np.round(values, 6)))
        assert distinct >= 0.8 * len(values), (distinct, len(values))

        first = reference_smoothness_loss(
            activations,
            reference_patch_units_3d(layout.cell_to_unit, (2, 2, 2), RADIUS_3D),
            d_unit,
        )
        far = reference_smoothness_loss(
            activations,
            reference_patch_units_3d(layout.cell_to_unit, (4, 5, 6), RADIUS_3D),
            d_unit,
        )
        assert first != pytest.approx(far, abs=1e-6)

    def test_a_missing_normalisation_is_rejected(self, fixture_3d):
        _, patch, d_unit, activations = fixture_3d
        correct = reference_smoothness_loss(activations, patch, d_unit)
        uncentred = 1.0 / (
            np.arange(d_unit.shape[0], dtype=np.float64) % 5 + 1.0
        )
        mutated = reference_smoothness_loss(
            activations, patch, uncentred.astype(np.float32)
        )
        assert abs(correct - mutated) > 1e-3

    @pytest.mark.parametrize("axis", [0, 2])
    def test_a_prior_that_forgets_an_axis_is_rejected(self, fixture_3d, axis):
        """The 2D-in-3D slip: a prior blind to the axis the field varies along.

        A field varying only along ``axis`` scores 0.449 (axis 0) or 0.440
        (axis 2) against the correct prior and 0.520 or 0.517 against one that
        drops that axis. A 2D prior applied to a 3D patch would make exactly
        this error, and a loose "less than 0.5" band cannot tell the two apart.
        """
        layout, patch, d_unit, _ = fixture_3d
        frequency = [(0, 0, 0), (0, 0, 0), (0, 0, 0)]
        modes = [tuple(1 if k == axis else 0 for k in range(3)),
                 tuple(2 if k == axis else 0 for k in range(3))]
        activations = smooth_field_activations_3d(
            layout, 48, seed=0, frequencies=[frequency[0], *modes]
        )
        correct = reference_smoothness_loss(activations, patch, d_unit)
        blind = reference_smoothness_loss(
            activations, patch, _axis_blind_d_unit(2 * RADIUS_3D + 1, axis)
        )
        assert correct < 0.5
        assert abs(correct - blind) > 0.05, (correct, blind)

    def test_the_three_metrics_disagree_with_each_other(self):
        priors = {m: reference_d_unit_3d(27, m) for m in ("linf", "l1", "l2")}
        assert not np.allclose(priors["linf"], priors["l1"], atol=1e-3)
        assert not np.allclose(priors["l1"], priors["l2"], atol=1e-3)
        assert not np.allclose(priors["linf"], priors["l2"], atol=1e-3)

    def test_a_perfect_cube_is_required(self):
        with pytest.raises(ValueError, match="not a perfect cube"):
            reference_d_unit_3d(121)


class TestTheThreeDimensionalHelpersAgainstHandDerivedValues:
    """Pins derived on paper, not read off any code."""

    def test_d_unit_3d_linf_side_3_matches_the_closed_form(self):
        """A 3x3x3 patch has 351 pairs.

        Per axis, the ordered offset pairs with ``|delta| <= 1`` are 3 equal and
        4 adjacent, 7 in all, so ``7**3 = 343`` ordered cell pairs have l-inf
        separation at most 1; removing the 27 identical ones and halving gives
        158 unordered pairs at separation 1. The other ``351 - 158 = 193`` pairs
        are at separation 2. The inverse distances are 1/2 and 1/3, so the
        centred, unit-normalised prior takes exactly two values.
        """
        near_pairs, far_pairs = 158, 193
        near, far = 1.0 / 2.0, 1.0 / 3.0
        mean = (near_pairs * near + far_pairs * far) / 351.0
        norm = np.sqrt(
            near_pairs * (near - mean) ** 2 + far_pairs * (far - mean) ** 2
        )
        expected_near = (near - mean) / norm
        expected_far = (far - mean) / norm

        d_unit = reference_d_unit_3d(27, "linf")
        assert d_unit.shape == (351,)
        assert int(np.isclose(d_unit, expected_near, atol=1e-6).sum()) == near_pairs
        assert int(np.isclose(d_unit, expected_far, atol=1e-6).sum()) == far_pairs
        # The first pair is cells (0,0,0) and (0,0,1): one step along axis 2.
        assert d_unit[0] == pytest.approx(expected_near, abs=1e-6)

    @pytest.mark.parametrize("distance", ["linf", "l1", "l2"])
    def test_d_unit_3d_is_centred_and_unit_norm(self, distance):
        d_unit = reference_d_unit_3d(125, distance).astype(np.float64)
        assert abs(float(d_unit.mean())) < 1e-6
        assert float(np.linalg.norm(d_unit)) == pytest.approx(1.0, abs=1e-6)

    @pytest.mark.parametrize(
        "distance,diagonal_separation",
        [("l1", 3.0), ("l2", float(np.sqrt(3.0)))],
    )
    def test_the_diagonal_pair_counts_all_three_axes(
        self, distance, diagonal_separation
    ):
        """Norm-free pin of the cross-axis term.

        In a 3x3x3 patch cell 0 is (0,0,0), cell 1 is (0,0,1), cell 2 is (0,0,2)
        and cell 13 is (1,1,1). Inverse distances are ``1/(s + 1)``; centring and
        normalising are affine, so the ratio of two differences cancels them:
        ``(d[0,13] - d[0,1]) / (d[0,2] - d[0,1])`` equals
        ``(v_diag - 1/2) / (1/3 - 1/2)`` with ``v_diag = 1/(s_diag + 1)``. That is
        1.5 for l1 (``s = 3``) and 0.80385 for l2 (``s = sqrt(3)``); a prior that
        dropped one axis would read 1.0 and 0.5147.
        """
        d_unit = reference_d_unit_3d(27, distance).astype(np.float64)
        step = d_unit[_pair_index(27, 0, 1)]
        two_steps = d_unit[_pair_index(27, 0, 2)]
        diagonal = d_unit[_pair_index(27, 0, 13)]
        expected = (1.0 / (diagonal_separation + 1.0) - 0.5) / (1.0 / 3.0 - 0.5)
        assert (diagonal - step) / (two_steps - step) == pytest.approx(
            expected, abs=1e-5
        )

    def test_the_patch_extractor_orders_axis_2_fastest(self):
        """Values pinned on a non-cubic grid where every axis has its own length.

        ``cell_to_unit = arange(5 * 6 * 7).reshape(5, 6, 7)`` stores
        ``(i0 * 6 + i1) * 7 + i2`` at ``[i0, i1, i2]``. The radius-1 block around
        ``(2, 3, 3)`` starts at ``(1, 2, 2)``: ``1 * 42 + 2 * 7 + 2 = 58`` and
        ends at ``(3, 4, 4)``: ``3 * 42 + 4 * 7 + 4 = 158``. A transposed axis
        changes the first element, the second, or the stride pattern.
        """
        cells = np.arange(5 * 6 * 7).reshape(5, 6, 7)
        block = reference_patch_units_3d(cells, (2, 3, 3), 1)
        assert block.shape == (27,)
        assert block[0] == 58
        assert block[1] == 59        # axis 2 is fastest: step 1
        assert block[3] == 65        # axis 1: stride 7
        assert block[9] == 100       # axis 0: stride 42
        assert block[-1] == 158
        np.testing.assert_array_equal(block, cells[1:4, 2:5, 2:5].reshape(-1))

    @pytest.mark.parametrize(
        "frequency,varying_axis", [((1, 0, 0), 0), ((0, 1, 0), 1), ((0, 0, 1), 2)]
    )
    def test_the_field_generator_varies_along_the_axis_its_frequency_names(
        self, frequency, varying_axis
    ):
        """On a (5, 6, 7) identity layout a single mode varies along ONE axis.

        The order of the frequency tuple is the order of ``grid_shape``; a
        generator that transposed it would vary along a different axis, which on
        a cubic grid would be invisible.
        """
        layout = types.SimpleNamespace(grid_shape=(5, 6, 7), perm=np.arange(210))
        field = smooth_field_activations_3d(
            layout, 3, seed=0, frequencies=(frequency,)
        )[0].reshape(5, 6, 7)
        for axis in range(3):
            spread = float(np.ptp(field, axis=axis).max())
            if axis == varying_axis:
                assert spread > 0.1, (axis, spread)
            else:
                assert spread < 1e-9, (axis, spread)


def _pair_index(patch_units, i, j):
    """Position of the pair ``(i, j)``, ``i < j``, in row-major pair order."""
    before = sum(patch_units - 1 - k for k in range(i))
    return before + (j - i - 1)
