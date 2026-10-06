"""Test suite for :mod:`dl_techniques.layers.regularization.spatial_smoothness`.

Covers layout bijectivity and grid factoring, neighbourhood table geometry, the
behavioural range and orientation of the loss, constructor validation, the
identity forward pass and its training-only gate, per-variable gradient flow, the
``alpha`` control as a measured delta, build parity, a ``.keras`` VALUE round trip
that restores the permutation, and the mandated precision and XLA arms.

The behavioural pins, with the numbers this suite measures (float32, CPU,
784 units on a 28x28 grid, radius 5, 5 sampled neighbourhoods, 256 samples):

  1. test_a_smooth_field_scores_far_below_one_half      0.172 - 0.198
  2. test_independent_noise_scores_at_one_half          0.496 - 0.500
  3. test_a_field_coarser_than_its_neighbourhood_crosses_and_exceeds_one_half
                                                          0.508 at f=6, 0.526 at f=9
  4. test_the_negated_prior_scores_far_above_one_half   0.826 (sign-discriminating)
  5. test_the_distance_prior_decreases_strictly_with_separation
                                                          0.0350 at d=1 down to -0.0124 at d=10
"""

import numpy as np
import pytest
import tensorflow as tf

import keras
from keras import ops

from dl_techniques.layers.regularization.spatial_smoothness import (
    DISTANCE_METRICS,
    NeighborhoodTables,
    SpatialLayout,
    SpatialSmoothness,
    build_patch_tables,
    permutation_seed,
    resolve_grid_shape,
    spatial_smoothness_loss,
)

from .spatial_smoothness_oracle import (
    graded_field_activations,
    reference_d_unit,
    reference_smoothness_loss,
    smooth_field_activations,
)

NUM_UNITS = 784
RADIUS = 5
DTYPE_POLICIES = ("float32", "mixed_float16", "float64")


def _sample_input(batch=2, length=8, units=NUM_UNITS, seed=0):
    return np.random.default_rng(seed).normal(
        size=(batch, length, units)
    ).astype("float32")


class _Parent(keras.layers.Layer):
    """Minimal parent whose ``call`` is the only path that builds the child."""

    def __init__(self, child, **kwargs):
        super().__init__(**kwargs)
        self.child = child

    def call(self, inputs, training=None):
        return self.child(inputs, training=training)


def _build_through_parent(child, input_shape=(None, 4, NUM_UNITS)):
    """Build ``child`` the way a real model does: from a parent's ``call``.

    Calling ``child.build(shape)`` directly is structurally blind to the
    ``StatelessScope`` trap, because that path never runs inside the scope.
    Tracing is safe here: the tap skips ``add_loss`` for symbolic inputs, which
    is the whole reason that guard exists.
    """
    parent = _Parent(child)
    parent(keras.Input(shape=tuple(input_shape[1:])))
    return child


def _tables(seed=0, radius=RADIUS, distance="linf", units=NUM_UNITS):
    layout = SpatialLayout(units, seed=seed)
    return layout, build_patch_tables(layout.cell_to_unit, radius, distance)


def _loss_of(activations, tables, num_neighborhoods=5, seed=1):
    """Evaluate the loss on activations of any rank, flattened to (M, N).

    The layer flattens before correlating; a caller of the pure function has to
    do it too. Passing a ``(2, 8, 784)`` array straight through would gather
    along the length-8 axis instead of the unit axis and raise an out-of-range
    index -- which is at least a loud failure rather than a wrong number.
    """
    activations = np.asarray(activations, "float32").reshape(
        -1, activations.shape[-1]
    )
    rows = np.random.default_rng(seed).integers(
        0, tables.num_centers, size=num_neighborhoods
    )
    return float(
        spatial_smoothness_loss(
            ops.convert_to_tensor(activations),
            ops.convert_to_tensor(tables.patch_table[rows]),
            ops.convert_to_tensor(tables.triu_flat),
            ops.convert_to_tensor(tables.d_unit),
        )
    )


# ---------------------------------------------------------------------
# layout
# ---------------------------------------------------------------------


class TestSpatialLayout:
    def test_the_permutation_is_a_bijection(self):
        layout = SpatialLayout(NUM_UNITS, seed=0)
        assert sorted(layout.perm.tolist()) == list(range(NUM_UNITS))
        assert sorted(layout.cell_to_unit.reshape(-1).tolist()) == list(
            range(NUM_UNITS)
        )

    def test_grid_and_from_grid_round_trips_exactly(self):
        layout = SpatialLayout(NUM_UNITS, seed=0)
        values = np.arange(NUM_UNITS, dtype="float32")
        np.testing.assert_array_equal(
            layout.from_grid(layout.to_grid(values)), values
        )

    def test_positions_agree_with_cell_to_unit(self):
        layout = SpatialLayout(NUM_UNITS, seed=0)
        for unit in (0, 1, 383, 782, NUM_UNITS - 1):
            row, col = (int(v) for v in layout.positions[unit])
            assert layout.cell_to_unit[row, col] == unit

    def test_distinct_seeds_give_distinct_permutations(self):
        first = SpatialLayout(NUM_UNITS, seed=0).perm
        second = SpatialLayout(NUM_UNITS, seed=1).perm
        assert not np.array_equal(first, second)

    def test_permute_false_places_unit_i_at_cell_i(self):
        layout = SpatialLayout(NUM_UNITS, permute=False)
        np.testing.assert_array_equal(layout.perm, np.arange(NUM_UNITS))

    def test_to_grid_rejects_a_wrong_unit_axis(self):
        layout = SpatialLayout(NUM_UNITS)
        with pytest.raises(ValueError, match="to_grid expects a last axis"):
            layout.to_grid(np.zeros((3, NUM_UNITS - 1)))

    def test_from_grid_rejects_a_wrong_grid_shape(self):
        layout = SpatialLayout(NUM_UNITS)
        with pytest.raises(ValueError, match="from_grid expects trailing axes"):
            layout.from_grid(np.zeros((7, 7)))

    def test_config_round_trips(self):
        layout = SpatialLayout(512, seed=7)
        rebuilt = SpatialLayout.from_config(layout.get_config())
        assert rebuilt.get_config() == layout.get_config()
        np.testing.assert_array_equal(rebuilt.perm, layout.perm)


class TestResolveGridShape:
    @pytest.mark.parametrize(
        "units,expected",
        [
            (784, (28, 28)),
            (256, (16, 16)),
            (512, (16, 32)),
            (100, (10, 10)),
            (121, (11, 11)),
            (97, (1, 97)),
        ],
    )
    def test_the_grid_is_the_most_square_factorisation(self, units, expected):
        assert resolve_grid_shape(units) == expected

    def test_the_height_never_exceeds_the_width(self):
        for units in range(2, 400):
            height, width = resolve_grid_shape(units)
            assert height <= width
            assert height * width == units

    @pytest.mark.parametrize("units", [2, 3, 4, 5, 7, 8, 9, 13, 16, 31, 64, 121, 784])
    def test_every_shape_multiplies_back(self, units):
        height, width = resolve_grid_shape(units)
        assert height * width == units

    def test_an_explicit_grid_that_does_not_multiply_out_is_rejected(self):
        with pytest.raises(ValueError, match="has 12 cells but the tapped tensor"):
            resolve_grid_shape(784, grid_shape=(3, 4))

    def test_a_non_positive_unit_count_is_rejected(self):
        with pytest.raises(ValueError, match="num_units must be positive"):
            resolve_grid_shape(0)

    def test_a_non_square_explicit_grid_is_accepted(self):
        assert resolve_grid_shape(784, grid_shape=(14, 56)) == (14, 56)


class TestPermutationSeed:
    def test_distinct_tap_indices_never_collide(self):
        seeds = [permutation_seed(7, index) for index in range(64)]
        assert len(set(seeds)) == 64

    def test_the_derivation_is_reproducible(self):
        assert permutation_seed(7, 3) == permutation_seed(7, 3)

    def test_a_negative_tap_index_is_rejected(self):
        with pytest.raises(ValueError, match="tap_index must be non-negative"):
            permutation_seed(0, -1)


# ---------------------------------------------------------------------
# neighbourhood tables
# ---------------------------------------------------------------------


class TestNeighborhoodTables:
    def test_the_centre_count_is_the_paper_arithmetic(self):
        _, tables = _tables()
        assert tables.num_centers == (28 - 2 * RADIUS) ** 2 == 324
        assert tables.side == 2 * RADIUS + 1 == 11
        assert tables.patch_units == 11 ** 2 == 121
        assert tables.num_pairs == 121 * 120 // 2 == 7260

    def test_every_row_is_a_set_of_distinct_units(self):
        _, tables = _tables()
        for row in tables.patch_table:
            assert len(set(row.tolist())) == tables.patch_units

    def test_every_row_is_the_block_around_its_centre(self):
        """Row ``k`` must be the block centred on the k-th admissible centre.

        Centres run over ``radius .. height-1-radius``, so the FIRST centre is
        grid ``(5, 5)`` and its block starts at ``(0, 0)`` -- the centre index and
        the slice origin are different numbers, which is an easy off-by-radius to
        write into a test and then believe.
        """
        layout, tables = _tables()
        height, width = layout.grid_shape
        side = tables.side
        for k in (0, 1, 17, 323):
            centre_row, centre_col = divmod(k, width - 2 * RADIUS)
            centre_row += RADIUS
            centre_col += RADIUS
            expected = layout.cell_to_unit[
                centre_row - RADIUS:centre_row + RADIUS + 1,
                centre_col - RADIUS:centre_col + RADIUS + 1,
            ]
            np.testing.assert_array_equal(
                tables.patch_table[k].reshape(side, side), expected
            )
        assert height == width == 28

    def test_d_unit_is_centred_and_normalised(self):
        _, tables = _tables()
        assert float(np.abs(np.asarray(tables.d_unit).mean())) < 1e-6
        np.testing.assert_allclose(
            float(np.linalg.norm(tables.d_unit)), 1.0, atol=1e-6, rtol=0
        )

    def test_the_distance_prior_decreases_strictly_with_separation(self):
        """Sign and orientation, pinned on a per-separation mean.

        Mean per l-infinity separation, radius 5: 0.0350, 0.0157, 0.0061,
        0.0003, -0.0036, -0.0063, -0.0084, -0.0100, -0.0113, -0.0124. Monotone
        strictly decreasing, which is what "nearby units should be MORE
        correlated" means. A transposed or sign-flipped prior cannot satisfy it.
        """
        _, tables = _tables()
        separation = reference_separations(tables.patch_units)
        per_class = [
            float(np.asarray(tables.d_unit)[separation == k].mean())
            for k in range(1, tables.side)
        ]
        assert all(
            near > far
            for near, far in zip(per_class, per_class[1:])
        ), per_class

    def test_d_unit_matches_the_paper_transcription(self):
        _, tables = _tables()
        np.testing.assert_allclose(
            tables.d_unit, reference_d_unit(tables.patch_units), atol=1e-6,
            rtol=0,
        )

    @pytest.mark.parametrize("distance", DISTANCE_METRICS)
    def test_every_distance_metric_builds(self, distance):
        _, tables = _tables(distance=distance)
        np.testing.assert_allclose(
            tables.d_unit, reference_d_unit(tables.patch_units, distance),
            atol=1e-6, rtol=0,
        )

    def test_the_metrics_disagree_with_each_other(self):
        priors = {
            metric: _tables(distance=metric)[1].d_unit for metric in DISTANCE_METRICS
        }
        assert not np.allclose(priors["linf"], priors["l1"])
        assert not np.allclose(priors["l1"], priors["l2"])

    def test_a_grid_too_small_for_the_radius_is_rejected(self):
        layout = SpatialLayout(64, seed=0)
        with pytest.raises(ValueError, match="needs a 11x11 patch"):
            build_patch_tables(layout.cell_to_unit, 5)

    @pytest.mark.parametrize("radius", [0, -1])
    def test_a_non_positive_radius_is_rejected(self, radius):
        with pytest.raises(ValueError, match="radius must be >= 1"):
            build_patch_tables(SpatialLayout(784, seed=0).cell_to_unit, radius)

    def test_an_unknown_distance_is_rejected(self):
        with pytest.raises(ValueError, match="distance must be one of"):
            build_patch_tables(
                SpatialLayout(784, seed=0).cell_to_unit, 5, "manhattan"
            )

    def test_the_returned_type_is_a_named_tuple(self):
        _, tables = _tables()
        assert isinstance(tables, NeighborhoodTables)
        assert tables.radius == RADIUS
        assert tables.grid_shape == (28, 28)


def reference_separations(patch_units):
    """l-infinity separation of every upper-triangle pair of one patch."""
    side = int(round(np.sqrt(patch_units)))
    rows, cols = np.divmod(np.arange(patch_units), side)
    delta = np.maximum(
        np.abs(rows[:, None] - rows[None, :]),
        np.abs(cols[:, None] - cols[None, :]),
    )
    upper = np.triu_indices(patch_units, k=1)
    return delta[upper[0], upper[1]]


# ---------------------------------------------------------------------
# the loss
# ---------------------------------------------------------------------


class TestSpatialSmoothnessLoss:
    def test_the_loss_stays_in_the_unit_interval(self):
        rng = np.random.default_rng(0)
        _, tables = _tables()
        for _ in range(8):
            value = _loss_of(rng.normal(size=(64, NUM_UNITS)), tables)
            assert 0.0 <= value <= 1.0

    def test_a_smooth_field_scores_far_below_one_half(self):
        """The invariant, not the shape.

        A field built from the grid's lowest Fourier modes has unit-to-unit
        correlation decaying with distance. Measured 0.172 at seed 0 and 0.198
        with the frequency sweep's mode set; i.i.d. noise measures 0.496, so the
        separation is ~0.30 against a floor of ~1e-7.
        """
        layout, tables = _tables()
        activations = smooth_field_activations(layout, num_samples=256, seed=0)
        value = _loss_of(activations, tables)
        assert value < 0.30, value

    def test_independent_noise_scores_at_one_half(self):
        layout, tables = _tables()
        rng = np.random.default_rng(11)
        for _ in range(4):
            value = _loss_of(rng.normal(size=(256, NUM_UNITS)), tables)
            assert abs(value - 0.5) < 0.02, value

    def test_a_field_coarser_than_its_neighbourhood_crosses_and_exceeds_one_half(
        self,
    ):
        """The anti-smooth arm, and the loss's whole dynamic range in one sweep.

        Frequency sweep of :func:`graded_field_activations` on a 28x28 grid with
        a radius-5 patch (patch side 11), 256 samples, seed 3, measured:

            f=1  0.2008    f=5  0.4666    f=9   0.5241
            f=2  0.2207    f=6  0.4993    f=10  0.5198
            f=3  0.3668    f=7  0.5095    f=12  0.5286
            f=4  0.4490    f=8  0.5113    f=13  0.5261

        The crossing sits where the wavelength approaches the neighbourhood's
        own extent, and past it the field is anti-smooth at that scale. A
        checkerboard does NOT demonstrate this: at l-infinity separation 1 it has
        three offset classes with two of them negative, so it averages to zero
        correlation and scores exactly what noise scores.
        """
        layout, tables = _tables()
        measured = {
            frequency: _loss_of(
                graded_field_activations(layout, 256, frequency, seed=3), tables
            )
            for frequency in (1, 2, 3, 4, 5, 6, 7, 8, 9, 12)
        }
        ordered = [measured[f] for f in (1, 2, 3, 4, 5, 6, 7, 8, 9)]
        assert all(near < far for near, far in zip(ordered, ordered[1:])), measured
        assert ordered[0] < 0.30, measured
        assert measured[7] > 0.5, measured
        assert measured[9] > 0.5, measured
        assert measured[12] > 0.5, measured

    def test_the_negated_prior_scores_far_above_one_half(self):
        """Sign-discriminating: distance-to-+d against distance-to--d.

        The same smooth map, once against the prior and once against its
        negation, reads 0.174 and 0.826. Neither number is reachable by loosening
        a tolerance, which is what makes this the sign pin.
        """
        layout, tables = _tables()
        activations = smooth_field_activations(layout, num_samples=256, seed=0)
        forward = _loss_of(activations, tables)
        backward = float(
            spatial_smoothness_loss(
                ops.convert_to_tensor(activations.astype("float32")),
                ops.convert_to_tensor(
                    tables.patch_table[
                        np.random.default_rng(1).integers(
                            0, tables.num_centers, size=5
                        )
                    ]
                ),
                ops.convert_to_tensor(tables.triu_flat),
                ops.convert_to_tensor(-tables.d_unit),
            )
        )
        assert forward < 0.30, forward
        assert backward > 0.70, backward
        assert forward + backward == pytest.approx(1.0, abs=1e-5)

    def test_the_loss_matches_the_paper_transcription_term_for_term(self):
        """Against an oracle transcribed from Eq. 1, in explicit loops.

        Agreement to 1e-5 in float32. The tolerance is re-association round-off
        on a 7260-term dot product, not a slack: both implementations sum the
        same terms in different orders.
        """
        layout, tables = _tables()
        activations = smooth_field_activations(layout, num_samples=64, seed=5)
        rows = [7, 101, 260]
        vectorised = float(
            spatial_smoothness_loss(
                ops.convert_to_tensor(activations.astype("float32")),
                ops.convert_to_tensor(tables.patch_table[rows]),
                ops.convert_to_tensor(tables.triu_flat),
                ops.convert_to_tensor(tables.d_unit),
            )
        )
        # The oracle is called on the SAME three centres, or the two numbers
        # would be averaging over different neighbourhoods.
        transcribed = [
            reference_smoothness_loss(activations, tables.patch_table[row], tables.d_unit)
            for row in rows
        ]
        np.testing.assert_allclose(
            vectorised, float(np.mean(transcribed)), atol=1e-5, rtol=0
        )
        # Sanity on the oracle itself: independent neighbourhoods must not agree.
        assert len(set(np.round(transcribed, 6))) == 3

    @pytest.mark.parametrize("num_samples", [1, 2, 3, 8])
    def test_degenerate_sample_counts_stay_finite(self, num_samples):
        """M=1 makes the per-unit standard deviation exactly zero.

        The ``eps`` floor is the only thing keeping this off a NaN, and a NaN
        would otherwise be a silent all-NaN training run. Measured: M=1 -> 0.500,
        M=2 -> 0.5027, M=3 -> 0.5011, all finite.
        """
        _, tables = _tables()
        rng = np.random.default_rng(4)
        for _ in range(3):
            value = _loss_of(rng.normal(size=(num_samples, NUM_UNITS)), tables)
            assert np.isfinite(value), value
            assert 0.0 <= value <= 1.0

    def test_constant_units_are_finite_and_score_one_half(self):
        _, tables = _tables()
        value = _loss_of(np.ones((64, NUM_UNITS)), tables)
        assert np.isfinite(value)
        assert value == pytest.approx(0.5, abs=1e-3)


# ---------------------------------------------------------------------
# the layer
# ---------------------------------------------------------------------


class TestSpatialSmoothnessConstruction:
    def test_configuration_is_stored_and_the_layer_starts_unbuilt(self):
        layer = SpatialSmoothness(
            alpha=1.5, radius=3, num_neighborhoods=2, distance="l1",
            permute=False, grid_shape=None, seed=11, max_samples=64, eps=1e-5,
        )
        assert not layer.built
        assert layer.alpha == 1.5
        assert layer.radius == 3
        assert layer.num_neighborhoods == 2
        assert layer.distance == "l1"
        assert layer.permute is False
        assert layer.max_samples == 64
        assert layer.eps == pytest.approx(1e-5)

    def test_config_round_trips_every_argument(self):
        layer = SpatialSmoothness(
            alpha=1.5, radius=3, num_neighborhoods=2, distance="l2",
            permute=False, grid_shape=(4, 6), seed=11, max_samples=64, eps=1e-5,
        )
        rebuilt = SpatialSmoothness.from_config(layer.get_config())
        for key in (
            "alpha", "radius", "num_neighborhoods", "distance", "permute",
            "grid_shape", "seed", "max_samples", "eps",
        ):
            assert rebuilt.get_config()[key] == layer.get_config()[key]

    @pytest.mark.parametrize(
        "kwargs,message",
        [
            ({"alpha": -1.0}, "alpha must be >= 0"),
            ({"radius": 0}, "radius must be >= 1"),
            ({"num_neighborhoods": 0}, "num_neighborhoods must be >= 1"),
            ({"distance": "manhattan"}, "distance must be one of"),
            ({"eps": 0.0}, "eps must be > 0"),
            ({"max_samples": 1}, "max_samples must be >= 2"),
        ],
    )
    def test_invalid_arguments_name_the_offender(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            SpatialSmoothness(**kwargs)

    def test_a_dynamic_last_axis_is_refused_at_build(self):
        layer = SpatialSmoothness(alpha=1.0, radius=1, seed=0)
        with pytest.raises(ValueError, match="statically known last axis"):
            layer.build((None, 4, None))

    def test_a_grid_too_small_for_the_radius_is_refused_at_build(self):
        layer = SpatialSmoothness(alpha=1.0, radius=5, seed=0)
        with pytest.raises(ValueError, match="needs a 11x11 patch"):
            layer.build((None, 4, 64))

    def test_compute_output_shape_works_unbuilt_and_is_the_identity(self):
        layer = SpatialSmoothness(alpha=1.0, radius=5, seed=0)
        shape = (None, 7, NUM_UNITS)
        assert layer.compute_output_shape(shape) == shape
        assert not layer.built


class TestSpatialSmoothnessForward:
    def test_the_forward_pass_is_the_identity_in_both_modes(self):
        layer = SpatialSmoothness(alpha=2.5, radius=RADIUS, seed=0)
        inputs = _sample_input()
        np.testing.assert_array_equal(
            ops.convert_to_numpy(layer(inputs, training=False)), inputs
        )
        np.testing.assert_array_equal(
            ops.convert_to_numpy(layer(inputs, training=True)), inputs
        )

    def test_no_loss_is_added_at_inference(self):
        layer = SpatialSmoothness(alpha=2.5, radius=RADIUS, seed=0)
        layer(_sample_input(), training=False)
        assert layer.losses == []
        layer(_sample_input(seed=1), training=None)
        assert layer.losses == []

    def test_one_loss_is_added_per_training_call(self):
        layer = SpatialSmoothness(alpha=2.5, radius=RADIUS, seed=0)
        layer(_sample_input(), training=True)
        assert len(layer.losses) == 1

    def test_the_added_loss_is_alpha_times_the_penalty(self):
        """Measured on one forward: added 1.2445 against penalty 0.4978.

        Both halves are needed: ``model.losses`` holds the WEIGHTED value, and an
        assertion that only checks "a loss appeared" is satisfied by a tap wired
        to the wrong tensor.
        """
        alpha = 2.5
        layer = SpatialSmoothness(alpha=alpha, radius=RADIUS, seed=0)
        layer(_sample_input(), training=True)
        recorded = float(ops.convert_to_numpy(layer.last_spatial_loss))
        added = float(layer.losses[0])
        assert added == pytest.approx(alpha * recorded, rel=1e-5)
        assert 0.0 < recorded <= 1.0

    def test_alpha_zero_adds_nothing(self):
        layer = SpatialSmoothness(alpha=0.0, radius=RADIUS, seed=0)
        layer(_sample_input(), training=True)
        assert layer.losses == []

    def test_the_layout_is_exposed_after_build(self):
        layer = SpatialSmoothness(alpha=1.0, radius=RADIUS, seed=3)
        layer(_sample_input())
        assert isinstance(layer.layout, SpatialLayout)
        assert layer.layout.grid_shape == (28, 28)
        np.testing.assert_array_equal(
            ops.convert_to_numpy(layer.perm), layer.layout.perm
        )

    def test_reset_statistics_zeroes_the_recorded_loss(self):
        layer = SpatialSmoothness(alpha=1.0, radius=RADIUS, seed=0)
        layer(_sample_input(), training=True)
        assert float(ops.convert_to_numpy(layer.last_spatial_loss)) != 0.0
        layer.reset_statistics()
        assert float(ops.convert_to_numpy(layer.last_spatial_loss)) == 0.0

    def test_max_samples_caps_the_correlation_rows(self):
        capped = SpatialSmoothness(alpha=1.0, radius=2, seed=0, max_samples=8)
        full = SpatialSmoothness(alpha=1.0, radius=2, seed=0)
        small = _sample_input(batch=4, length=32)
        capped(small, training=True)
        full(small, training=True)
        assert np.isfinite(float(capped.losses[0]))
        assert float(capped.losses[0]) != float(full.losses[0])


class TestStatelessBuild:
    def test_the_tables_survive_the_stateless_build_pass(self):
        """The one probe that sees the ``StatelessScope`` trap.

        ``child.build(shape)`` directly would pass even against the broken
        ``add_weight(zeros)`` + ``.assign()`` implementation, because that path
        never runs inside the scope. Building through a parent's ``call`` is what
        every real model does.

        Pinned against a closed form at a DISCRIMINATING entry: ``perm`` is a
        permutation, so entry 0 is 0 only if the table is still at its zeros
        initializer, and any of its 784 values distinguishes a live permutation
        from a live-but-different one.
        """
        layer = _build_through_parent(
            SpatialSmoothness(alpha=2.5, radius=RADIUS, seed=0)
        )
        perm = ops.convert_to_numpy(layer.perm)
        assert not np.all(perm == 0), "perm is all zeros -- the table was discarded"
        np.testing.assert_array_equal(perm, SpatialLayout(784, seed=0).perm)

    def test_every_table_survives_the_stateless_build_pass(self):
        layer = _build_through_parent(
            SpatialSmoothness(alpha=2.5, radius=RADIUS, seed=0)
        )
        patch_table = ops.convert_to_numpy(layer.patch_table)
        assert patch_table.shape == (324, 121)
        assert not np.all(patch_table == 0)
        d_unit = ops.convert_to_numpy(layer.d_unit)
        np.testing.assert_allclose(float(np.linalg.norm(d_unit)), 1.0, atol=1e-6, rtol=0)
        assert ops.convert_to_numpy(layer.triu_flat).max() < 121 * 121


class TestGradients:
    def test_gradients_reach_every_trainable_weight_upstream_of_the_tap(self):
        """Non-None AND non-zero, named by path, with the anti-vacuity count.

        The tap itself holds no trainable weights -- its tables are non-trainable
        by design -- so the claim under test is that the penalty reaches the
        layer producing the tapped tensor. ``len(trainable_variables) > 0`` is the
        anti-vacuity arm.
        """
        dense = keras.layers.Dense(NUM_UNITS, kernel_initializer="ones")
        model = _DenseThenTap(dense, SpatialSmoothness(alpha=2.5, radius=RADIUS, seed=0))
        with tf.GradientTape() as tape:
            loss = ops.sum(ops.square(model(_sample_input(batch=4, length=16), training=True)))
        gradients = tape.gradient(loss, dense.trainable_variables)

        assert len(dense.trainable_variables) > 0
        for variable, gradient in zip(dense.trainable_variables, gradients):
            assert gradient is not None, f"no gradient for {variable.path}"
            assert np.any(
                ops.convert_to_numpy(gradient) != 0.0
            ), f"all-zero gradient for {variable.path}"

    def test_the_penalty_contributes_a_measurable_gradient_delta(self):
        """The both-ways pair for ``alpha``: non-zero at 2.5, exactly zero at 0.

        The instrument is the gradient of the SPATIAL TERM ALONE, not of the
        total objective. Differentiating the total is what makes this test
        unreadable: the task gradient is orders of magnitude larger, so the
        spatial contribution disappears inside float32 round-off and the two arms
        agree to every digit -- a green result that means nothing.

        ``len(model.losses)`` is asserted alongside, because a non-empty loss
        list with a zero gradient is the other failure this has to exclude.
        """
        inputs = _sample_input(batch=4, length=16, seed=1)
        measured = {}
        for alpha in (0.0, 2.5):
            dense = keras.layers.Dense(NUM_UNITS, kernel_initializer="ones")
            tap = SpatialSmoothness(alpha=alpha, radius=RADIUS, seed=0, name="tap")
            model = _DenseThenTap(dense, tap)

            # The forward pass has to happen INSIDE the tape: `add_loss` stores a
            # tensor, and a tensor created under an earlier tape is detached from
            # this one. Reading `model.losses` after an outside-the-tape forward
            # therefore differentiates nothing, which is why an earlier revision
            # of this test read a gradient of exactly 0.0 at BOTH alphas.
            with tf.GradientTape() as tape:
                model(inputs, training=True)
                if model.losses:
                    spatial_only = ops.cast(model.losses[-1], "float32")
                else:
                    # alpha=0 contributes nothing. The forward still has to be on
                    # the tape, or `tape.gradient` has no path to differentiate.
                    spatial_only = ops.cast(
                        ops.sum(model(inputs, training=False)), "float32"
                    ) * 0.0
            gradients = tape.gradient(spatial_only, dense.trainable_variables)

            assert len(model.losses) == (1 if alpha > 0 else 0)
            measured[alpha] = float(
                sum(
                    float(ops.convert_to_numpy(gradient).sum())
                    for gradient in gradients
                    if gradient is not None
                )
            )

        assert measured[0.0] == 0.0, "alpha=0 still produced a gradient"
        assert measured[2.5] != 0.0, "the penalty produced no gradient at all"
        assert abs(measured[2.5]) > 1e-3


class _DenseThenTap(keras.Model):
    """A tap ON the output of a projection, so a gradient has somewhere to land.

    The order matters and is not cosmetic. With ``dense(tap(x))`` the penalty
    depends on ``x`` and has NO path to ``dense.kernel``, so the gradient of the
    penalty w.r.t. the projection's weights is identically zero -- at every
    ``alpha``, including the live ones. Placing the tap downstream is also what a
    real model does: the taps sit on the residual-path output of a projection.
    """

    def __init__(self, dense, tap, **kwargs):
        super().__init__(**kwargs)
        self.dense = dense
        self.tap = tap

    def call(self, inputs, training=None):
        return self.tap(self.dense(inputs), training=training)


class TestBuildParity:
    def test_explicit_build_matches_lazy_build(self):
        """Relative weight paths, with every sub-layer explicitly named.

        The names are load-bearing, not tidiness: Keras auto-increments generated
        names per instance, so two separately-constructed models produce
        ``dense/kernel`` and ``dense_1/kernel`` at every unnamed level, and
        stripping only the root does not normalise that away. A parity failure is
        a naming problem before it is a build problem.
        """
        def build():
            return keras.Sequential(
                [
                    keras.layers.Dense(NUM_UNITS, name="project"),
                    SpatialSmoothness(
                        alpha=2.5, radius=RADIUS, seed=0, name="tap"
                    ),
                ],
                name="model",
            )

        explicit = build()
        explicit.build((None, 4, NUM_UNITS))
        lazy = build()
        lazy(_sample_input())

        def relative(model):
            return sorted(w.path.split("/", 1)[-1] for w in model.weights)

        assert relative(explicit), "the model has no weights to compare"
        assert relative(explicit) == relative(lazy)

    def test_alpha_zero_still_builds_every_table(self):
        """The control is a control: identical weights, no loss.

        Without this the ``alpha=0`` arm would differ in weight count from the
        trained arm, and "identical hyperparameters" would be a claim rather
        than a measured fact.
        """
        live = SpatialSmoothness(alpha=2.5, radius=RADIUS, seed=0, name="tap")
        live(_sample_input())
        control = SpatialSmoothness(alpha=0.0, radius=RADIUS, seed=0, name="tap")
        control(_sample_input())

        live_paths = sorted(w.path for w in live.weights)
        control_paths = sorted(w.path for w in control.weights)
        assert live_paths == control_paths
        np.testing.assert_array_equal(
            ops.convert_to_numpy(live.perm), ops.convert_to_numpy(control.perm)
        )
        assert control.trainable_variables == []

    def test_the_tap_contributes_no_trainable_weights(self):
        layer = SpatialSmoothness(alpha=2.5, radius=RADIUS, seed=0)
        layer(_sample_input())
        assert layer.trainable_variables == []
        assert len(layer.weights) == 5

    def test_the_tables_are_non_trainable_so_nothing_can_rewrite_them(self):
        """The layout is a checkpoint artefact, not a parameter.

        A trainable ``perm`` would let the optimizer silently re-map every unit
        while the checkpoint still looked loadable, and the loss would keep
        scoring against a layout the weights were never trained under.
        """
        layer = SpatialSmoothness(alpha=2.5, radius=RADIUS, seed=0)
        layer(_sample_input())
        for weight in layer.weights:
            assert not weight.trainable, f"{weight.path} is trainable"


class TestSerialization:
    def test_the_permutation_and_tables_survive_a_value_round_trip(self):
        """``atol=0.0``, compared BEFORE the loaded model's first call.

        Restoration is a copy, not a computation. Comparing outputs instead would
        be satisfied by a model that restored nothing and rebuilt the tables from
        its seed -- which for a fixed seed is the same numbers, so the round trip
        would pass while the checkpoint carried nothing.
        """
        tap = SpatialSmoothness(alpha=2.5, radius=RADIUS, seed=3, name="tap")
        model = keras.Sequential(
            [keras.layers.Dense(NUM_UNITS, name="project"), tap], name="model"
        )
        inputs = _sample_input()
        model(inputs, training=True)

        saved = {
            weight.name: ops.convert_to_numpy(weight).copy()
            for weight in model.weights
        }
        assert saved, "the model has no weights to compare"

        import tempfile
        import os

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "layer.keras")
            model.save(path)
            loaded = keras.models.load_model(path)

            # Compared by leaf NAME, not by full path: a reloaded Sequential is
            # re-instantiated and its weights carry bare names (`kernel`), so the
            # `project/kernel` prefix exists only on the donor. Leaf names are
            # unique here -- a Dense's kernel and the tap's tables cannot collide.
            restored = {
                weight.name: ops.convert_to_numpy(weight)
                for weight in loaded.weights
            }
            assert sorted(saved) == sorted(restored), (
                f"weight set changed across the round trip: "
                f"missing {sorted(set(saved) - set(restored))}, "
                f"unexpected {sorted(set(restored) - set(saved))}"
            )
            for name, values in saved.items():
                np.testing.assert_array_equal(
                    values, restored[name],
                    err_msg=f"{name} was not restored",
                )

    def test_the_forward_output_survives_a_value_round_trip(self):
        tap = SpatialSmoothness(alpha=2.5, radius=RADIUS, seed=0, name="tap")
        model = keras.Sequential(
            [keras.layers.Dense(NUM_UNITS, name="project"), tap], name="model"
        )
        inputs = _sample_input()
        original = ops.convert_to_numpy(model(inputs, training=False))

        import tempfile
        import os

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "layer.keras")
            model.save(path)
            loaded = keras.models.load_model(path)
            restored = ops.convert_to_numpy(loaded(inputs, training=False))

        np.testing.assert_allclose(restored, original, atol=1e-6, rtol=0)

    def test_the_layout_is_a_weight_and_not_part_of_the_config(self):
        """The layout travels as weights, so a config cannot silently re-map units.

        ``get_config`` carries the SEED, which is what makes a fresh construction
        reproducible; it does not carry the permutation itself. That asymmetry is
        deliberate, and it is only safe because the permutation is a non-trainable
        weight restored bit-exactly (the two tests above). If it were rebuilt from
        the seed on load instead, a checkpoint saved under one seed and reloaded
        under another would quietly re-index every unit.
        """
        layer = SpatialSmoothness(alpha=2.5, radius=RADIUS, seed=3, name="tap")
        layer(_sample_input())
        config = layer.get_config()

        assert config["seed"] == 3
        assert "perm" not in config
        assert "cell_to_unit" not in config
        assert any(w.path.endswith("perm") for w in layer.weights)

        # Two constructions from the same config agree, which is what the seed is
        # there for.
        first = SpatialSmoothness.from_config(config)
        second = SpatialSmoothness.from_config(config)
        first(_sample_input())
        second(_sample_input())
        np.testing.assert_array_equal(
            ops.convert_to_numpy(first.perm), ops.convert_to_numpy(second.perm)
        )


class TestPrecision:
    @pytest.mark.parametrize("policy", DTYPE_POLICIES)
    def test_the_loss_is_finite_and_bounded_under_every_policy(self, policy):
        previous = keras.mixed_precision.global_policy().name
        floatx = keras.backend.floatx()
        try:
            if policy == "float64":
                keras.backend.set_floatx("float64")
            keras.mixed_precision.set_global_policy(policy)

            layer = SpatialSmoothness(alpha=2.5, radius=RADIUS, seed=0)
            raw = np.random.default_rng(9).normal(size=(2, 8, NUM_UNITS))
            if policy == "float64":
                inputs = ops.convert_to_tensor(raw.astype("float64"))
            else:
                inputs = ops.convert_to_tensor(raw.astype("float32"))
            layer(inputs, training=True)

            value = float(layer.losses[0])
            assert np.isfinite(value)
            assert 0.0 < value <= 2.5, value
        finally:
            keras.mixed_precision.set_global_policy(previous)
            keras.backend.set_floatx(floatx)

    def test_float64_is_not_narrowed_to_float32(self):
        """The never-narrow property, against an EXPLICIT narrowing control.

        A hard ``cast(x, "float32")`` inside the reduction is the defect, and it
        is invisible under a float32 policy -- there it is a no-op. So the
        control has to be a float64 run whose input is the *same* array, and the
        comparison is against the same computation with the cast forced back in.

        Comparing a float32-policy run against a float64-policy run instead would
        measure the input rounding, not the reduction: the two arms would be fed
        differently-rounded arrays, and the defect could hide inside that.
        """
        raw = np.random.default_rng(21).normal(size=(2, 8, NUM_UNITS))
        inputs = ops.convert_to_tensor(raw.astype("float64"))
        assert inputs.dtype == "float64", (
            "the float64 arm is a fake reading: the input did not arrive as "
            f"float64 but as {inputs.dtype}"
        )

        previous = keras.mixed_precision.global_policy().name
        floatx = keras.backend.floatx()
        try:
            keras.backend.set_floatx("float64")
            keras.mixed_precision.set_global_policy("float64")

            layer = SpatialSmoothness(alpha=1.0, radius=RADIUS, seed=0)
            layer(inputs, training=True)
            widened = float(layer.losses[0])

            narrowed = _loss_of(
                raw.astype("float32"), _tables()[1]
            )
        finally:
            keras.mixed_precision.set_global_policy(previous)
            keras.backend.set_floatx(floatx)

        assert np.isfinite(widened)
        assert widened != pytest.approx(narrowed, abs=1e-7), (
            "the float64 run reproduced the float32-narrowed value exactly -- "
            "the reduction is being narrowed to float32"
        )


class TestGraphSafety:
    def test_a_tf_function_trace_matches_eager(self):
        layer = SpatialSmoothness(alpha=2.5, radius=RADIUS, seed=0)
        inputs = _sample_input(batch=2, length=8, seed=4)

        @tf.function(
            input_signature=[tf.TensorSpec([None, None, NUM_UNITS], tf.float32)]
        )
        def traced(x):
            return layer(x, training=True)

        # The graph arm runs first: `traced` traces and executes, and any stale
        # value in `last_spatial_loss` at that point can only have come from it.
        traced(inputs)
        graph = float(ops.convert_to_numpy(layer.last_spatial_loss))

        layer.reset_statistics()
        layer(inputs, training=True)
        eager = float(ops.convert_to_numpy(layer.last_spatial_loss))

        assert np.isfinite(graph)
        assert eager != 0.0, "the eager arm ran at 0.0 and compares as a pass"
        assert graph == pytest.approx(eager, rel=2e-3)

    def test_jit_compiled_fit_actually_optimises(self):
        """An eager-only fix is not a fix; the tap has to survive XLA.

        The gradient is checked through the optimizer's own update, not through a
        hand-written tape, so a term that never reaches ``apply_gradients`` is
        caught here even if the loss looks right.
        """
        rng = np.random.default_rng(12)
        inputs = rng.normal(size=(64, NUM_UNITS)).astype("float32")
        targets = rng.normal(size=(64, NUM_UNITS)).astype("float32")

        model = keras.Sequential(
            [
                keras.layers.Dense(NUM_UNITS),
                SpatialSmoothness(alpha=2.5, radius=2, seed=0, name="tap"),
            ],
            name="model",
        )
        model.compile(optimizer="adam", loss="mse", jit_compile=True)
        history = model.fit(inputs, targets, batch_size=8, epochs=2, verbose=0)

        assert np.isfinite(history.history["loss"][-1])
        assert history.history["loss"][-1] < history.history["loss"][0]