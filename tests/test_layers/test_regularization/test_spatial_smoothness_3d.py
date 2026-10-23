"""3D (``grid_dim=3``) tests for :mod:`dl_techniques.layers.regularization.spatial_smoothness`.

RED-FIRST. This module was written BEFORE the production change it describes
(plan-2026-10-23T141429-390f3815, step 1), against the design in decisions
D-002 to D-006, and it fails test by test on the 2D-only module: the names and
keywords it exercises (``_factor_3d_cubic``, ``grid_dim=`` on
``resolve_grid_shape`` / ``SpatialLayout`` / ``SpatialSmoothness``,
``NeighborhoodTables.grid_dim``, 3-axis ``build_patch_tables``) do not exist yet.
New names are reached through the module object or by keyword inside each test,
so the file collects and every test reports its own failure.

The 2D layer is pinned separately and GREEN in
``test_spatial_smoothness.py::TestTwoDimensionalIsByteIdentical``; nothing here
repeats it. The 3D oracles live in ``spatial_smoothness_oracle.py`` and are
independent of the implementation (explicit loops, no ``np.unravel_index``).

One axis order, everywhere (D-003): ``grid_shape == cell_to_unit.shape ==
(axis 0, axis 1, axis 2)``, row-major, axis 2 fastest, no depth/height/width
naming. A centre count ``(a - 2r)(b - 2r)(c - 2r)`` is symmetric under axis
permutation, so shape and count asserts cannot see a reorder; the VALUE pins
below (a non-cubic ``(5, 6, 7)`` grid, a ``(7, 8, 14)`` grid) can.

Numbers, derived with the independent oracle before any 3D production code
existed:

  * per-separation mean of the radius-2 l-infinity prior (side 5, 125 units):
    0.0246, 0.0044, -0.0057, -0.0118 at separations 1..4 (strictly decreasing);
  * smooth 3D field on an (8, 8, 8) layout, radius 2, 64 samples: loss 0.137 to
    0.326 over all 64 centres, mean 0.197; a field indexed by ANOTHER
    permutation scores 0.495 to 0.504 at every centre tried.
"""

import functools

import numpy as np
import pytest

import keras

from dl_techniques.layers.regularization import (
    spatial_smoothness as spatial_smoothness_module,
)
from dl_techniques.layers.regularization.spatial_smoothness import (
    DISTANCE_METRICS,
    NeighborhoodTables,
    SpatialLayout,
    SpatialSmoothness,
    build_patch_tables,
    resolve_grid_shape,
    spatial_smoothness_loss,
)
from tests.layer_naming import PublicAutoName

from .spatial_smoothness_oracle import (
    reference_d_unit_3d,
    reference_patch_units_3d,
    reference_smoothness_loss,
    smooth_field_activations_3d,
)

DTYPE_POLICIES = ("float32", "mixed_float16", "float64")

#: Pinned by D-002: ``n -> (h, w, d)`` with ``h <= w <= d``, ``h*w*d == n``,
#: minimising ``d - h``, ties to the larger ``h``. Re-derived by running the
#: rule, not carried from the design document (which pins 784 to (7, 7, 16)).
FACTOR_PINS = {
    343: (7, 7, 7),
    512: (8, 8, 8),
    256: (4, 8, 8),
    784: (7, 8, 14),
    1000: (10, 10, 10),
    997: (1, 1, 997),
    1331: (11, 11, 11),
}


def _layout_3d(num_units, seed=0, **kwargs):
    return SpatialLayout(num_units, grid_dim=3, seed=seed, **kwargs)


def _factor_3d_cubic(n):
    """Reach the private factoring function through the module object."""
    return spatial_smoothness_module._factor_3d_cubic(n)


def _brute_force_factor(n):
    """Independent oracle for D-002: enumerate every ordered triplet.

    No integer cube root, no float power: every divisor ``h``, every divisor
    ``w >= h`` of ``n / h``, ``d`` the quotient; keep ``d >= w``; minimise
    ``d - h`` and break ties toward the larger ``h``.
    """
    best_key, best = None, None
    for h in range(1, n + 1):
        if n % h:
            continue
        rest = n // h
        for w in range(h, rest + 1):
            if rest % w:
                continue
            d = rest // w
            if d < w:
                break
            key = (d - h, -h)
            if best_key is None or key < best_key:
                best_key, best = key, (h, w, d)
    return best


def _lexicographic_centres(shape, radius):
    """Admissible centres, axis 0 slowest, written as three explicit loops."""
    return [
        (c0, c1, c2)
        for c0 in range(radius, shape[0] - radius)
        for c1 in range(radius, shape[1] - radius)
        for c2 in range(radius, shape[2] - radius)
    ]


class _Parent(PublicAutoName, keras.layers.Layer):
    """Parent whose ``call`` is the only path that builds the child."""

    def __init__(self, child, **kwargs):
        super().__init__(**kwargs)
        self.child = child

    def call(self, inputs, training=None):
        return self.child(inputs, training=training)


def _build_through_parent(child, units):
    """Build ``child`` the way a model does: inside a parent's ``call``."""
    parent = _Parent(child)
    parent(keras.Input(shape=(4, units)))
    return child


# ---------------------------------------------------------------------
# D-002: the integer factoring
# ---------------------------------------------------------------------


class TestCubicFactoring:
    @pytest.mark.parametrize("n,expected", sorted(FACTOR_PINS.items()))
    def test_the_pinned_widths_factor_as_designed(self, n, expected):
        assert _factor_3d_cubic(n) == expected

    def test_the_documents_784_pin_is_not_the_most_cubic_factoring(self):
        """784 is (7, 8, 14), not the (7, 7, 16) the design document pinned.

        ``d - h`` is 7 for (7, 8, 14) and 9 for (7, 7, 16); the first wins.
        """
        assert _factor_3d_cubic(784) == (7, 8, 14)
        assert _factor_3d_cubic(784) != (7, 7, 16)

    def test_the_rule_agrees_with_a_brute_force_oracle_for_every_width(self):
        mismatches = [
            (n, _factor_3d_cubic(n), _brute_force_factor(n))
            for n in range(1, 2001)
            if _factor_3d_cubic(n) != _brute_force_factor(n)
        ]
        assert not mismatches, mismatches[:5]

    def test_the_result_is_sorted_and_multiplies_back(self):
        for n in (1, 2, 12, 100, 360, 997, 1024, 1999):
            h, w, d = _factor_3d_cubic(n)
            assert h <= w <= d
            assert h * w * d == n

    def test_every_perfect_cube_is_its_own_cube_root(self):
        """A float cube root rounds some cubes down; the integer rule cannot."""
        bad = [k for k in range(1, 201) if _factor_3d_cubic(k ** 3) != (k, k, k)]
        assert not bad, bad


# ---------------------------------------------------------------------
# D-004: resolve_grid_shape(num_units, grid_shape, grid_dim)
# ---------------------------------------------------------------------


class TestResolveGridShape3D:
    @pytest.mark.parametrize("n", sorted(FACTOR_PINS))
    def test_auto_factoring_uses_the_cubic_rule(self, n):
        assert resolve_grid_shape(n, grid_dim=3) == FACTOR_PINS[n]

    def test_an_explicit_triple_is_accepted_in_the_order_given(self):
        assert resolve_grid_shape(336, (8, 6, 7), grid_dim=3) == (8, 6, 7)

    def test_a_pair_is_refused_when_the_rank_is_three(self):
        """The rank is the stated ``grid_dim``, never inferred from the length."""
        with pytest.raises(ValueError, match="grid_dim"):
            resolve_grid_shape(512, (16, 32), grid_dim=3)

    def test_a_cell_count_mismatch_names_both_counts(self):
        with pytest.raises(ValueError, match="has 64 cells but the tapped tensor"):
            resolve_grid_shape(100, (4, 4, 4), grid_dim=3)

    def test_a_non_positive_entry_is_refused(self):
        with pytest.raises(ValueError, match="must be positive"):
            resolve_grid_shape(64, (0, 8, 8), grid_dim=3)

    def test_num_units_must_be_positive_in_3d_too(self):
        with pytest.raises(ValueError, match="num_units must be positive"):
            resolve_grid_shape(0, grid_dim=3)

    @pytest.mark.parametrize("grid_dim", [1, 4, 0, -3, "3"])
    def test_grid_dim_outside_two_and_three_is_refused_by_value(self, grid_dim):
        with pytest.raises(ValueError, match="grid_dim"):
            resolve_grid_shape(64, grid_dim=grid_dim)


# ---------------------------------------------------------------------
# D-003: SpatialLayout on a 3D grid
# ---------------------------------------------------------------------


class TestSpatialLayout3D:
    def test_the_grid_is_the_cubic_factoring(self):
        layout = _layout_3d(784)
        assert layout.grid_shape == (7, 8, 14)
        assert layout.cell_to_unit.shape == (7, 8, 14)
        assert layout.positions.shape == (784, 3)

    def test_dtypes_match_the_2d_layout(self):
        layout = _layout_3d(784)
        assert layout.perm.dtype == np.int32
        assert layout.positions.dtype == np.int32
        assert layout.cell_to_unit.dtype == np.int32

    def test_the_permutation_draw_is_the_2d_draw(self):
        """The same seed draws the same permutation in either rank."""
        np.testing.assert_array_equal(
            _layout_3d(784, seed=5).perm, SpatialLayout(784, seed=5).perm
        )

    def test_the_layout_is_a_bijection(self):
        layout = _layout_3d(784, seed=2)
        assert sorted(layout.perm.tolist()) == list(range(784))
        assert sorted(layout.cell_to_unit.reshape(-1).tolist()) == list(range(784))

    def test_positions_index_cell_to_unit_back_to_the_unit(self):
        layout = _layout_3d(784, seed=2)
        p0, p1, p2 = (layout.positions[:, k] for k in range(3))
        np.testing.assert_array_equal(
            layout.cell_to_unit[p0, p1, p2], np.arange(784)
        )

    def test_positions_are_row_major_with_axis_2_fastest(self):
        """VALUE pin on a grid whose three axes have three different lengths.

        ``permute=False`` puts unit ``i`` at cell ``i``; on a ``(5, 6, 7)`` grid
        cell ``i`` is ``(i // 42, (i // 7) % 6, i % 7)``. Unit 58 is (1, 2, 2),
        unit 100 is (2, 2, 2), the last is (4, 5, 6). A transposed or reordered
        decomposition moves at least one of these.
        """
        layout = SpatialLayout(210, (5, 6, 7), permute=False, grid_dim=3)
        assert layout.positions[58].tolist() == [1, 2, 2]
        assert layout.positions[100].tolist() == [2, 2, 2]
        assert layout.positions[209].tolist() == [4, 5, 6]
        assert layout.positions[1].tolist() == [0, 0, 1]
        assert layout.positions[7].tolist() == [0, 1, 0]
        assert layout.positions[42].tolist() == [1, 0, 0]
        assert layout.cell_to_unit[1, 2, 2] == 58

    def test_to_grid_places_each_unit_at_its_cell(self):
        layout = _layout_3d(784, seed=1)
        values = np.random.default_rng(0).normal(size=(3, 784))
        grid = layout.to_grid(values)
        assert grid.shape == (3, 7, 8, 14)
        for cell in ((0, 0, 0), (3, 4, 9), (6, 7, 13)):
            unit = layout.cell_to_unit[cell]
            np.testing.assert_array_equal(grid[(slice(None),) + cell], values[:, unit])

    def test_from_grid_is_the_exact_inverse_with_leading_axes(self):
        layout = _layout_3d(784, seed=1)
        values = np.random.default_rng(0).normal(size=(2, 3, 784))
        np.testing.assert_array_equal(
            layout.from_grid(layout.to_grid(values)), values
        )

    @pytest.mark.parametrize("bad_shape", [(8, 14), (7, 8, 13), (8, 7, 14)])
    def test_from_grid_refuses_a_wrong_trailing_shape(self, bad_shape):
        layout = _layout_3d(784)
        with pytest.raises(ValueError, match="from_grid expects trailing axes"):
            layout.from_grid(np.zeros(bad_shape))

    def test_to_grid_rejects_a_wrong_unit_axis(self):
        with pytest.raises(ValueError, match="to_grid expects a last axis"):
            _layout_3d(784).to_grid(np.zeros((3, 783)))

    def test_an_explicit_triple_sets_the_grid(self):
        layout = SpatialLayout(336, (8, 6, 7), seed=1, grid_dim=3)
        assert layout.grid_shape == (8, 6, 7)
        assert layout.cell_to_unit.shape == (8, 6, 7)

    def test_a_pair_is_refused_for_a_3d_layout(self):
        with pytest.raises(ValueError, match="grid_dim"):
            SpatialLayout(512, (16, 32), grid_dim=3)

    def test_the_config_writes_grid_dim_and_round_trips(self):
        layout = _layout_3d(784, seed=4)
        config = layout.get_config()
        assert config["grid_dim"] == 3
        assert tuple(config["grid_shape"]) == (7, 8, 14)
        rebuilt = SpatialLayout.from_config(config)
        assert rebuilt.grid_shape == (7, 8, 14)
        np.testing.assert_array_equal(rebuilt.perm, layout.perm)
        np.testing.assert_array_equal(rebuilt.positions, layout.positions)

    def test_a_2d_layout_writes_grid_dim_two(self):
        assert SpatialLayout(784, seed=0).get_config()["grid_dim"] == 2


# ---------------------------------------------------------------------
# D-005: build_patch_tables on a 3D grid
# ---------------------------------------------------------------------


# Plain cached functions rather than fixtures: a fixture that raises reports an
# ERROR at setup for every test that uses it, while these report each test's own
# FAILED. The returned arrays are only read.


@functools.lru_cache(maxsize=None)
def _tables_8_8_8():
    layout = _layout_3d(512, seed=0)
    return layout, build_patch_tables(layout.cell_to_unit, 2, "linf")


@functools.lru_cache(maxsize=None)
def _identity_5_6_7():
    """``cell_to_unit = arange(210).reshape(5, 6, 7)``, radius 2, 6 centres."""
    layout = SpatialLayout(210, (5, 6, 7), permute=False, grid_dim=3)
    return layout, build_patch_tables(layout.cell_to_unit, 2, "linf")


class TestPatchTables3D:
    def test_the_arithmetic_for_a_cubic_grid(self):
        _, tables = _tables_8_8_8()
        assert tables.grid_dim == 3
        assert tables.grid_shape == (8, 8, 8)
        assert tables.side == 5
        assert tables.patch_units == 5 ** 3 == 125
        assert tables.num_pairs == 125 * 124 // 2 == 7750
        assert tables.num_centers == (8 - 4) ** 3 == 64
        assert tables.patch_table.shape == (64, 125)
        assert tables.patch_table.dtype == np.int32
        assert tables.triu_flat.dtype == np.int32
        assert tables.d_unit.dtype == np.float32
        assert tables.d_unit.shape == (7750,)

    def test_a_2d_table_reports_grid_dim_two(self):
        tables = build_patch_tables(SpatialLayout(784, seed=0).cell_to_unit, 5)
        assert isinstance(tables, NeighborhoodTables)
        assert tables.grid_dim == 2

    def test_the_pair_index_runs_over_the_upper_triangle_in_row_major_order(
        self
    ):
        _, tables = _tables_8_8_8()
        assert tables.triu_flat[0] == 1                    # (0, 1)
        assert tables.triu_flat[1] == 2                    # (0, 2)
        assert tables.triu_flat[-1] == 123 * 125 + 124     # (123, 124)

    def test_every_row_is_a_set_of_distinct_units(self):
        _, tables = _tables_8_8_8()
        for row in tables.patch_table:
            assert len(set(row.tolist())) == 125

    def test_every_row_is_the_block_around_its_centre_in_lexicographic_order(
        self
    ):
        """Row ``k`` is the cube around the k-th centre, axis 0 slowest.

        The expectation is the oracle's three explicit loops, so a centre order
        or axis order that differs from the lexicographic one shows up as a
        different set of unit ids, not as a different shape.
        """
        layout, tables = _tables_8_8_8()
        centres = _lexicographic_centres(layout.grid_shape, 2)
        assert len(centres) == tables.num_centers
        for k in (0, 1, 4, 16, 37, 63):
            np.testing.assert_array_equal(
                tables.patch_table[k],
                reference_patch_units_3d(layout.cell_to_unit, centres[k], 2),
                err_msg=f"row {k} (centre {centres[k]})",
            )

    def test_a_non_cubic_grid_pins_the_axis_order_by_value(self):
        """``(5, 6, 7)``, radius 2, six centres, cell ``(i0, i1, i2)`` = ``i0*42 + i1*7 + i2``.

        Centres in lexicographic order: (2,2,2) (2,2,3) (2,2,4) (2,3,2) (2,3,3)
        (2,3,4). A block starts at centre minus 2 and ends at centre plus 2:
        first elements 0, 1, 2, 7, 8, 9 and last elements 200, 201, 202, 207,
        208, 209. Within row 0 the strides are 1 (axis 2), 7 (axis 1), 42
        (axis 0). A reversed axis order changes the first/last sets; a
        transposed one changes the strides.
        """
        _, tables = _identity_5_6_7()
        assert tables.num_centers == 1 * 2 * 3 == 6
        assert tables.patch_table[:, 0].tolist() == [0, 1, 2, 7, 8, 9]
        assert tables.patch_table[:, -1].tolist() == [200, 201, 202, 207, 208, 209]
        row = tables.patch_table[0]
        assert row[1] == 1       # axis 2 is fastest
        assert row[5] == 7       # axis 1: stride 7
        assert row[25] == 42     # axis 0: stride 42

    def test_d_unit_is_centred_and_normalised(self):
        _, tables = _tables_8_8_8()
        assert abs(float(np.asarray(tables.d_unit, "float64").mean())) < 1e-6
        np.testing.assert_allclose(
            float(np.linalg.norm(tables.d_unit)), 1.0, atol=1e-6, rtol=0
        )

    @pytest.mark.parametrize("distance", DISTANCE_METRICS)
    def test_every_metric_matches_the_independent_transcription(self, distance):
        layout = _layout_3d(512, seed=0)
        tables = build_patch_tables(layout.cell_to_unit, 2, distance)
        np.testing.assert_allclose(
            tables.d_unit, reference_d_unit_3d(tables.patch_units, distance),
            atol=1e-6, rtol=0,
        )

    def test_the_metrics_disagree_with_each_other(self):
        layout = _layout_3d(512, seed=0)
        priors = {
            metric: build_patch_tables(layout.cell_to_unit, 2, metric).d_unit
            for metric in DISTANCE_METRICS
        }
        assert not np.allclose(priors["linf"], priors["l1"], atol=1e-3)
        assert not np.allclose(priors["l1"], priors["l2"], atol=1e-3)
        assert not np.allclose(priors["linf"], priors["l2"], atol=1e-3)

    def test_the_distance_prior_decreases_strictly_with_separation(
        self
    ):
        """Per-separation means 0.0246, 0.0044, -0.0057, -0.0118 (separation 1..4).

        Measured on the oracle prior; the table's must be monotone the same way.
        """
        _, tables = _tables_8_8_8()
        cells = np.array(
            [(a, b, c) for a in range(5) for b in range(5) for c in range(5)]
        )
        separation = np.abs(cells[:, None, :] - cells[None, :, :]).max(axis=-1)
        upper = np.triu_indices(125, k=1)
        separation = separation[upper]
        per_class = [
            float(np.asarray(tables.d_unit)[separation == k].mean())
            for k in range(1, 5)
        ]
        assert all(near > far for near, far in zip(per_class, per_class[1:])), per_class

    @pytest.mark.parametrize("axis", [0, 1, 2])
    def test_one_short_axis_is_refused_even_when_the_total_is_large(self, axis):
        """Per-axis rejection (claims check item 6): 4*9*9 = 324 >= 125 cells.

        A total-size check passes all three shapes; only a per-axis check sees
        that the cube does not fit. The message follows the 2D wording and names
        the offending axis.
        """
        shape = [9, 9, 9]
        shape[axis] = 4
        cells = np.arange(int(np.prod(shape))).reshape(shape)
        with pytest.raises(ValueError, match="needs a 5x5x5 patch") as raised:
            build_patch_tables(cells, 2)
        assert f"axis {axis}" in str(raised.value)

    @pytest.mark.parametrize("shape", [(30,), (5, 5, 5, 5)])
    def test_only_two_or_three_axes_are_accepted(self, shape):
        with pytest.raises(ValueError, match="2 or 3"):
            build_patch_tables(np.zeros(shape, dtype="int32"), 1)

    @pytest.mark.parametrize("radius", [0, -1])
    def test_a_non_positive_radius_is_still_refused(self, radius):
        with pytest.raises(ValueError, match="radius must be >= 1"):
            build_patch_tables(_layout_3d(512).cell_to_unit, radius)

    def test_the_loss_matches_the_independent_transcription_term_for_term(
        self
    ):
        """Implementation tables against the oracle's own patches and prior.

        512 units on (8, 8, 8), radius 2, smooth field of 64 samples. The oracle
        side uses ``reference_patch_units_3d`` and ``reference_d_unit_3d`` --
        nothing from the tables -- so a wrong patch, a wrong centre order or a
        wrong prior fails here even though the vectorised loss itself is the
        untouched dimension-free function. Rows 3, 17, 40 are the centres
        (2,2,5), (3,2,3), (4,4,2).
        """
        layout, tables = _tables_8_8_8()
        activations = smooth_field_activations_3d(layout, num_samples=64, seed=5)
        rows = [3, 17, 40]
        vectorised = float(
            spatial_smoothness_loss(
                keras.ops.convert_to_tensor(activations.astype("float32")),
                keras.ops.convert_to_tensor(tables.patch_table[rows]),
                keras.ops.convert_to_tensor(tables.triu_flat),
                keras.ops.convert_to_tensor(tables.d_unit),
            )
        )
        centres = _lexicographic_centres(layout.grid_shape, 2)
        d_unit = reference_d_unit_3d(125, "linf")
        transcribed = [
            reference_smoothness_loss(
                activations,
                reference_patch_units_3d(layout.cell_to_unit, centres[row], 2),
                d_unit,
            )
            for row in rows
        ]
        np.testing.assert_allclose(
            vectorised, float(np.mean(transcribed)), atol=1e-5, rtol=0
        )
        assert len(set(np.round(transcribed, 6))) == 3


# ---------------------------------------------------------------------
# D-006: the SpatialSmoothness layer with grid_dim
# ---------------------------------------------------------------------


class TestSpatialSmoothness3DConstruction:
    def test_grid_dim_is_stored(self):
        layer = SpatialSmoothness(alpha=1.0, radius=2, seed=0, grid_dim=3)
        assert layer.grid_dim == 3
        assert SpatialSmoothness(alpha=1.0, radius=2, seed=0).grid_dim == 2

    @pytest.mark.parametrize("grid_dim", [1, 4, 0, "3"])
    def test_a_bad_grid_dim_names_the_offender(self, grid_dim):
        with pytest.raises(ValueError, match="grid_dim must be"):
            SpatialSmoothness(grid_dim=grid_dim)

    def test_the_config_carries_grid_dim_and_round_trips(self):
        layer = SpatialSmoothness(
            alpha=1.5, radius=2, num_neighborhoods=3, distance="l2",
            grid_shape=(8, 8, 8), seed=11, grid_dim=3,
        )
        config = layer.get_config()
        assert config["grid_dim"] == 3
        rebuilt = SpatialSmoothness.from_config(config)
        assert rebuilt.grid_dim == 3
        assert rebuilt.get_config() == config

    def test_a_2d_layer_writes_grid_dim_two(self):
        assert SpatialSmoothness(alpha=1.0, seed=0).get_config()["grid_dim"] == 2

    def test_a_pair_grid_shape_is_refused_for_a_3d_layer(self):
        # The same arguments with a triple are fine; the keyword itself is not
        # what is refused (an unknown-keyword error would also say "grid_dim").
        assert SpatialSmoothness(
            alpha=1.0, radius=1, seed=0, grid_shape=(8, 8, 8), grid_dim=3
        ).grid_dim == 3
        with pytest.raises(ValueError, match="grid_dim"):
            layer = SpatialSmoothness(
                alpha=1.0, radius=1, seed=0, grid_shape=(16, 32), grid_dim=3
            )
            layer.build((None, 4, 512))

    @pytest.mark.parametrize("axis", [0, 1, 2])
    def test_a_short_axis_is_refused_at_build_naming_the_axis(self, axis):
        shape = [9, 9, 9]
        shape[axis] = 4
        with pytest.raises(ValueError, match="needs a 5x5x5 patch") as raised:
            layer = SpatialSmoothness(
                alpha=1.0, radius=2, seed=0, grid_shape=tuple(shape), grid_dim=3
            )
            layer.build((None, 4, int(np.prod(shape))))
        assert f"axis {axis}" in str(raised.value)

    def test_784_units_at_radius_4_fail_on_the_7_cell_axis(self):
        """784 factors to (7, 8, 14): 784 >= 9**3 = 729 but axis 0 has 7 < 9."""
        with pytest.raises(ValueError, match="needs a 9x9x9 patch") as raised:
            layer = SpatialSmoothness(alpha=1.0, radius=4, seed=0, grid_dim=3)
            layer.build((None, 4, 784))
        assert "axis 0" in str(raised.value)


class TestSpatialSmoothness3DBuild:
    def test_the_tables_reach_the_weights(self):
        """Forgetting to forward ``grid_dim`` leaves a layer silently 2D."""
        layer = _build_through_parent(
            SpatialSmoothness(alpha=1.0, radius=2, seed=0, grid_dim=3), 512
        )
        layout = _layout_3d(512, seed=0)
        assert tuple(layer.patch_table.shape) == (64, 125)
        assert tuple(layer.triu_flat.shape) == (7750,)
        assert tuple(layer.d_unit.shape) == (7750,)
        np.testing.assert_array_equal(
            keras.ops.convert_to_numpy(layer.perm), layout.perm
        )
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(layer.d_unit),
            reference_d_unit_3d(125, "linf"), atol=1e-6, rtol=0,
        )
        centres = _lexicographic_centres(layout.grid_shape, 2)
        table = keras.ops.convert_to_numpy(layer.patch_table)
        for k in (0, 21, 63):
            np.testing.assert_array_equal(
                table[k],
                reference_patch_units_3d(layout.cell_to_unit, centres[k], 2),
            )

    def test_a_non_cubic_width_builds_the_lexicographic_cubes(self):
        """784 -> (7, 8, 14), radius 3: 1 * 2 * 8 = 16 centres of 343 units."""
        layer = _build_through_parent(
            SpatialSmoothness(alpha=1.0, radius=3, seed=0, grid_dim=3), 784
        )
        layout = _layout_3d(784, seed=0)
        table = keras.ops.convert_to_numpy(layer.patch_table)
        assert table.shape == (16, 343)
        centres = _lexicographic_centres((7, 8, 14), 3)
        assert centres[0] == (3, 3, 3) and centres[-1] == (3, 4, 10)
        for k in (0, 7, 15):
            np.testing.assert_array_equal(
                table[k],
                reference_patch_units_3d(layout.cell_to_unit, centres[k], 3),
            )

    def test_there_are_five_weights_in_the_pinned_order_and_int64_tables(self):
        layer = _build_through_parent(
            SpatialSmoothness(alpha=1.0, radius=2, seed=0, grid_dim=3), 512
        )
        assert len(layer.weights) == 5
        assert [w.path.rsplit("/", 1)[-1] for w in layer.weights] == [
            "perm", "patch_table", "triu_flat", "d_unit", "last_spatial_loss",
        ]
        assert [
            table.dtype for table in (layer.perm, layer.patch_table, layer.triu_flat)
        ] == ["int64"] * 3
        assert layer.trainable_variables == []

    def test_a_smooth_field_scores_low_and_another_layout_scores_one_half(self):
        """The tap scores the layout it was built with.

        The oracle loss over all 64 centres of the seed-0 smooth field spans
        0.137 to 0.326, so ANY draw of 5 centres averages below 0.35. The same
        values indexed by another permutation score 0.495 to 0.504 per centre.
        """
        layout = _layout_3d(512, seed=0)
        field = smooth_field_activations_3d(layout, num_samples=64, seed=5)
        inputs = field.astype("float32").reshape(2, 32, 512)

        aligned = SpatialSmoothness(alpha=1.0, radius=2, num_neighborhoods=5,
                                    seed=0, grid_dim=3)
        aligned(inputs, training=True)
        low = float(keras.ops.convert_to_numpy(aligned.last_spatial_loss))

        other = SpatialSmoothness(alpha=1.0, radius=2, num_neighborhoods=5,
                                  seed=1, grid_dim=3)
        other(inputs, training=True)
        high = float(keras.ops.convert_to_numpy(other.last_spatial_loss))

        assert 0.10 < low < 0.35, low
        assert abs(high - 0.5) < 0.05, high

    def test_the_forward_pass_is_the_identity(self):
        layer = SpatialSmoothness(alpha=1.0, radius=2, seed=0, grid_dim=3)
        inputs = np.random.default_rng(0).normal(size=(2, 6, 512)).astype("float32")
        np.testing.assert_array_equal(
            keras.ops.convert_to_numpy(layer(inputs, training=True)), inputs
        )

    def test_the_tables_survive_a_value_round_trip(self, tmp_path):
        """``.keras`` save and load, ``atol=0``, compared before the first call."""
        tap = SpatialSmoothness(alpha=2.5, radius=2, seed=3, grid_dim=3, name="tap")
        model = keras.Sequential(
            [keras.layers.Dense(512, name="project"), tap], name="model"
        )
        model(np.random.default_rng(0).normal(size=(2, 8, 512)).astype("float32"),
              training=True)
        saved = {
            w.name: keras.ops.convert_to_numpy(w).copy() for w in model.weights
        }

        path = str(tmp_path / "layer3d.keras")
        model.save(path)
        loaded = keras.models.load_model(path)
        restored = {
            w.name: keras.ops.convert_to_numpy(w) for w in loaded.weights
        }
        assert sorted(saved) == sorted(restored)
        for name, values in saved.items():
            np.testing.assert_array_equal(values, restored[name], err_msg=name)
        assert loaded.layers[-1].grid_dim == 3

    @pytest.mark.parametrize("policy", DTYPE_POLICIES)
    def test_the_loss_is_finite_and_bounded_under_every_policy(self, policy):
        previous = keras.mixed_precision.global_policy().name
        floatx = keras.backend.floatx()
        try:
            if policy == "float64":
                keras.backend.set_floatx("float64")
            keras.mixed_precision.set_global_policy(policy)
            layer = SpatialSmoothness(alpha=2.5, radius=2, seed=0, grid_dim=3)
            raw = np.random.default_rng(9).normal(size=(2, 8, 512))
            inputs = keras.ops.convert_to_tensor(
                raw.astype("float64" if policy == "float64" else "float32")
            )
            layer(inputs, training=True)
            value = float(layer.losses[0])
            assert np.isfinite(value)
            assert 0.0 < value <= 2.5, value
        finally:
            keras.mixed_precision.set_global_policy(previous)
            keras.backend.set_floatx(floatx)

    def test_a_jit_compiled_fit_optimises(self):
        rng = np.random.default_rng(12)
        inputs = rng.normal(size=(64, 512)).astype("float32")
        targets = rng.normal(size=(64, 512)).astype("float32")
        model = keras.Sequential(
            [
                keras.layers.Dense(512),
                SpatialSmoothness(alpha=2.5, radius=2, seed=0, grid_dim=3, name="tap"),
            ],
            name="model",
        )
        model.compile(optimizer="adam", loss="mse", jit_compile=True)
        history = model.fit(inputs, targets, batch_size=8, epochs=2, verbose=0)
        assert np.isfinite(history.history["loss"][-1])
        assert history.history["loss"][-1] < history.history["loss"][0]


class TestSpatialSmoothness3DConfigAndSign:
    """Config key set, legacy loading and the 3D sign probe (plan step 3)."""

    def test_the_config_key_set_is_exactly_the_old_keys_plus_grid_dim(self):
        keys = set(SpatialSmoothness(alpha=1.0, seed=0, grid_dim=3).get_config())
        assert keys == {
            "name", "trainable", "dtype",
            "alpha", "radius", "num_neighborhoods", "distance", "permute",
            "grid_shape", "seed", "max_samples", "eps", "grid_dim",
        }

    def test_a_config_without_grid_dim_builds_as_2d_even_with_a_3d_sibling(self):
        """The pre-3D config has no key; the default must be 2, not the last seen."""
        three = SpatialSmoothness(alpha=1.0, radius=2, seed=0, grid_dim=3)
        legacy = dict(three.get_config())
        del legacy["grid_dim"]
        legacy["grid_shape"] = None
        layer = SpatialSmoothness.from_config(legacy)
        assert layer.grid_dim == 2
        _build_through_parent(layer, 512)
        # 512 factors to (16, 32) in 2D: (16-4)*(32-4) = 336 centres of 25 units.
        assert tuple(layer.patch_table.shape) == (336, 25)
        assert layer.layout.grid_shape == (16, 32)
        assert layer.get_config()["grid_dim"] == 2

    def test_the_negated_prior_scores_far_above_one_half(self):
        """Sign-discriminating: distance-to-+d against distance-to--d, in 3D.

        512 units on (8, 8, 8), radius 2 (64 centres), the smooth 3D field of
        ``smooth_field_activations_3d`` (256 samples, seed 0), five fixed centres
        drawn with ``default_rng(1)``. MEASURED on the real implementation before
        any bound was written:

            forward (+d_unit)  0.2002      backward (-d_unit)  0.7998

        and over centre draws seeds 1, 2, 3: forward 0.2002 / 0.2066 / 0.1701,
        backward 0.7998 / 0.7934 / 0.8299 (the layout seed does not move them:
        the field is generated in the layout's own unit order). The pair is run
        twice, once on the tables ``build_patch_tables`` returns and once on the
        weights a built ``SpatialSmoothness(grid_dim=3)`` holds, so a flipped sign
        in either place turns it red.

        On a larger grid, 1000 units on (10, 10, 10) with radius 4 (only 8
        centres), the same pair reads 0.3311 / 0.6689 (and 0.3255 / 0.6745,
        0.3256 / 0.6744 for two more centre draws): still on the right sides of
        one half but too close to the 0.30 / 0.70 bands to assert, so it is
        recorded here only.
        """
        layout = _layout_3d(512, seed=0)
        tables = build_patch_tables(layout.cell_to_unit, 2, "linf")
        layer = _build_through_parent(
            SpatialSmoothness(alpha=1.0, radius=2, seed=0, grid_dim=3), 512
        )
        activations = smooth_field_activations_3d(
            layout, num_samples=256, seed=0
        ).astype("float32")
        rows = np.random.default_rng(1).integers(0, tables.num_centers, size=5)

        def pair(patch_table, triu_flat, d_unit):
            def loss(sign):
                return float(
                    spatial_smoothness_loss(
                        keras.ops.convert_to_tensor(activations),
                        keras.ops.convert_to_tensor(patch_table[rows]),
                        keras.ops.convert_to_tensor(triu_flat),
                        keras.ops.convert_to_tensor(sign * d_unit),
                    )
                )
            return loss(1.0), loss(-1.0)

        sources = {
            "tables": (tables.patch_table, tables.triu_flat, tables.d_unit),
            "layer": tuple(
                keras.ops.convert_to_numpy(w)
                for w in (layer.patch_table, layer.triu_flat, layer.d_unit)
            ),
        }
        for name, (patch_table, triu_flat, d_unit) in sources.items():
            forward, backward = pair(patch_table, triu_flat, d_unit)
            assert forward < 0.30, (name, forward)
            assert backward > 0.70, (name, backward)
            assert forward + backward == pytest.approx(1.0, abs=1e-5), name
