"""Test suite for :mod:`dl_techniques.layers.regularization.gaussian_readout`.

Covers the FWHM-to-sigma derivation, the kernel normalisations, grid resolution,
the identity-free forward pass over non-square grids, and the one claim that
matters most here: the in-graph blur and a NumPy reference must agree, because
two producers of the same number is the shape that drifts.

The mandated pins, with the numbers this suite measures (float32, CPU, 784 units
on a 28x28 grid, FWHM 2.0 at 1.0 spacing):

  1. test_a_delta_input_spreads_to_its_neighbours_and_no_further
  2. test_the_measured_fwhm_matches_the_requested_one   2.00 grid cells
  3. test_the_in_graph_blur_matches_the_numpy_reference  max |delta| < 1e-5
  4. test_the_blur_follows_the_grid_and_not_the_unit_axis
"""

import numpy as np
import pytest
import tensorflow as tf

import keras
from keras import ops

from dl_techniques.layers.regularization.gaussian_readout import (
    DEFAULT_FWHM,
    DEFAULT_UNIT_SPACING,
    PADDING_MODES,
    GaussianReadout,
    gaussian_kernel_1d,
    gaussian_kernel_2d,
    gaussian_sigma,
)
from dl_techniques.layers.regularization.spatial_smoothness import SpatialLayout
from tests.test_layers.test_regularization.spatial_smoothness_oracle import (
    smooth_field_activations,
)

NUM_UNITS = 784


def _numpy_blur(activations, layout, sigma):
    """Reference blur: scatter to the grid, filter, gather back."""
    from scipy import ndimage

    grid = layout.to_grid(np.asarray(activations, dtype=np.float64))
    filtered = ndimage.gaussian_filter(
        grid, sigma=(0.0, sigma, sigma), mode="nearest"
    )
    return layout.from_grid(filtered)


def _measured_fwhm(profile):
    """Full width at half maximum of a 1-D profile, by apex-relative interpolation.

    The ``>`` comparison is load-bearing. At the paper's FWHM of 2.0 with unit
    spacing the sigma is 0.8493, which makes ``exp(-1 / 2 sigma^2)`` land on
    exactly 0.5 -- so the taps one cell out sit ON the half-maximum rather than
    clearly above or below it, and a ``>=`` test includes them and reports a
    width of 3 for a kernel of true width 2.
    """
    peak_index = int(np.argmax(profile))
    peak = float(profile[peak_index])
    half = peak / 2.0

    def crossing(step):
        index = peak_index
        # Just below half, so a tap sitting exactly ON the half-maximum still
        # advances the walk. At the paper's FWHM of 2.0 the one-cell-out tap is
        # exactly 0.5 of the peak (exp(-1/2 sigma^2) = 0.5000 at sigma = 0.8493),
        # and the comparison cannot be written either way: `>` reads a width of 0
        # and `>=` reads a width of 3.
        threshold = half * (1.0 - 1e-6)
        while 0 <= index + step < len(profile) and profile[index + step] > threshold:
            index += step
        neighbour = index + step
        if not (0 <= neighbour < len(profile)):
            return float(index)
        here, there = float(profile[index]), float(profile[neighbour])
        # If the walk stopped ON the half-maximum, that sample IS the crossing.
        # Interpolating from it to the next one would report the far edge of the
        # interval instead, which at this kernel width is a whole cell out.
        if abs(here - half) <= 1e-6 * half:
            return float(index)
        if there == here:
            return float(neighbour)
        return index + (there - half) / (there - here) * step

    return float(crossing(1) - crossing(-1))


class TestGaussianSigma:
    def test_the_paper_setting_gives_the_documented_sigma(self):
        """FWHM 2.0 at 1.0 spacing -> sigma 0.8493.

        2.3548 is the FWHM of a Gaussian in sigmas; the division is what makes a
        width and a spacing commensurable.
        """
        assert gaussian_sigma(2.0, 1.0) == pytest.approx(0.8493, abs=1e-4)

    def test_sigma_scales_linearly_with_spacing_inverse(self):
        assert gaussian_sigma(4.0, 1.0) == pytest.approx(
            2.0 * gaussian_sigma(2.0, 1.0)
        )
        assert gaussian_sigma(2.0, 2.0) == pytest.approx(
            gaussian_sigma(2.0, 1.0) / 2.0
        )

    @pytest.mark.parametrize(
        "fwhm,spacing", [(0.0, 1.0), (-1.0, 1.0), (2.0, 0.0), (2.0, -1.0)]
    )
    def test_non_positive_inputs_are_rejected(self, fwhm, spacing):
        with pytest.raises(ValueError):
            gaussian_sigma(fwhm, spacing)


class TestGaussianKernel:
    def test_the_1d_kernel_sums_to_one(self):
        kernel = gaussian_kernel_1d(7, 1.0)
        assert float(kernel.sum()) == pytest.approx(1.0, abs=1e-12)

    def test_the_2d_kernel_sums_to_one(self):
        kernel = gaussian_kernel_2d(7, 1.0)
        assert float(kernel.sum()) == pytest.approx(1.0, abs=1e-12)

    def test_the_2d_kernel_is_separable(self):
        kernel = gaussian_kernel_2d(7, 1.3)
        line = gaussian_kernel_1d(7, 1.3)
        np.testing.assert_allclose(kernel, np.outer(line, line), atol=1e-15, rtol=0)

    def test_the_kernel_is_centred_and_symmetric(self):
        kernel = gaussian_kernel_1d(9, 1.2)
        np.testing.assert_allclose(kernel, kernel[::-1], atol=1e-15, rtol=0)
        assert int(np.argmax(kernel)) == 4

    @pytest.mark.parametrize("size", [0, 2, 4, -3])
    def test_an_even_or_non_positive_kernel_size_is_rejected(self, size):
        with pytest.raises(ValueError, match="positive odd integer"):
            gaussian_kernel_1d(size, 1.0)


class TestGaussianReadoutConstruction:
    def test_the_defaults_are_the_papers_configuration(self):
        layer = GaussianReadout()
        assert layer.fwhm == DEFAULT_FWHM == 2.0
        assert layer.unit_spacing == DEFAULT_UNIT_SPACING == 1.0
        assert layer.sigma == pytest.approx(0.8493, abs=1e-4)
        assert layer.kernel_size == 2 * int(np.ceil(3 * layer.sigma)) + 1 == 7

    def test_configuration_is_stored_and_the_layer_starts_unbuilt(self):
        layer = GaussianReadout(
            fwhm=3.0, unit_spacing=1.5, permute=False, grid_shape=(4, 6),
            seed=5, kernel_size=5, padding_mode="zeros",
        )
        assert not layer.built
        assert layer.fwhm == 3.0
        assert layer.unit_spacing == 1.5
        assert layer.kernel_size == 5
        assert layer.padding_mode == "zeros"

    def test_config_round_trips_every_argument(self):
        layer = GaussianReadout(
            fwhm=3.0, unit_spacing=1.5, permute=False, grid_shape=(4, 6),
            seed=5, kernel_size=5, padding_mode="zeros",
        )
        rebuilt = GaussianReadout.from_config(layer.get_config())
        assert rebuilt.get_config() == layer.get_config()

    @pytest.mark.parametrize(
        "kwargs,message",
        [
            ({"kernel_size": 4}, "positive odd integer"),
            ({"padding_mode": "reflect"}, "padding_mode must be one of"),
            ({"fwhm": 0.0}, "fwhm must be > 0"),
        ],
    )
    def test_invalid_arguments_name_the_offender(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            GaussianReadout(**kwargs)

    @pytest.mark.parametrize("mode", PADDING_MODES)
    def test_every_padding_mode_is_constructible(self, mode):
        layer = GaussianReadout(padding_mode=mode, seed=0)
        assert layer.padding_mode == mode

    def test_compute_output_shape_works_unbuilt(self):
        layer = GaussianReadout()
        shape = (None, 7, NUM_UNITS)
        assert layer.compute_output_shape(shape) == shape
        assert not layer.built


class TestGaussianReadoutForward:
    def _input(self, seed=0, units=NUM_UNITS, batch=2, length=6):
        return np.random.default_rng(seed).normal(
            size=(batch, length, units)
        ).astype("float32")

    def test_the_shape_is_preserved_and_the_output_is_finite(self):
        layer = GaussianReadout(seed=0)
        inputs = self._input()
        output = ops.convert_to_numpy(layer(inputs, training=False))
        assert output.shape == inputs.shape
        assert np.all(np.isfinite(output))

    def test_a_constant_field_passes_through_unchanged(self):
        """A normalised kernel over an edge-replicated grid preserves the DC term.

        This is the cheapest possible check that the kernel sums to one AND that
        the grid bookkeeping is not scrambling units. It is also the test that
        pins ``padding_mode='nearest'``: under Keras' zero-padded ``'same'`` the
        interior survives at 2.5 while the border decays to 1.835, because mass
        leaves the array. A constant field has to come back untouched.
        """
        layer = GaussianReadout(seed=0)
        inputs = np.full((1, 4, NUM_UNITS), 2.5, dtype="float32")
        np.testing.assert_allclose(
            ops.convert_to_numpy(layer(inputs, training=False)), inputs,
            atol=1e-4, rtol=0,
        )

    def test_a_delta_input_spreads_to_its_neighbours_and_no_further(self):
        """The positive liveness arm: the blur must actually move something.

        A dead readout returns its input and satisfies every equality below; this
        asserts the response is NON-zero where it should be, which an identity
        cannot.

        The peak is about 0.22, NOT 1.0: the kernel is normalised over its whole
        7x7 support, so the centre tap holds only a fifth of the mass. What must
        hold is MASS conservation -- the grid still sums to 1 -- and monotone
        falloff away from the source.
        """
        layout = SpatialLayout(NUM_UNITS, seed=0)
        layer = GaussianReadout(permute=True, seed=0)
        unit = 300
        inputs = _delta(NUM_UNITS, unit)

        output = ops.convert_to_numpy(layer(inputs, training=False))[0, 0]
        assert 0.0 < output[unit] < 1.0

        grid = layout.to_grid(output)
        row, col = (int(v) for v in layout.positions[unit])
        assert float(grid.sum()) == pytest.approx(1.0, abs=1e-4)
        forward = grid[row, col:col + 5]
        backward = grid[row, col::-1][:5]
        assert all(near > far for near, far in zip(forward, forward[1:]))
        assert all(near > far for near, far in zip(backward, backward[1:]))
        assert float(grid[row, col + 4]) < 1e-3

    def test_the_measured_fwhm_matches_the_requested_one(self):
        """The kernel width is a claim about the OUTPUT, not about the config.

        Checked against the continuous-Gaussian width ``2.3548 * sigma`` rather
        than read off ``layer.sigma``: the discrete 7x7 kernel is a sampled,
        renormalised truncation of the analytic Gaussian, so its profile is not
        exactly it and the bound has to carry that difference.
        """
        layout = SpatialLayout(NUM_UNITS, seed=0)
        layer = GaussianReadout(fwhm=2.0, unit_spacing=1.0, permute=True, seed=0)

        centre_unit = int(layout.cell_to_unit[14, 14])
        inputs = _delta(NUM_UNITS, centre_unit)

        output = ops.convert_to_numpy(layer(inputs, training=False))[0, 0]
        grid = layout.to_grid(output)
        profile = grid[14, 14 - 5:14 + 6]

        measured = _measured_fwhm(profile)
        expected = 2.0 * np.sqrt(2.0 * np.log(2.0)) * layer.sigma
        assert measured == pytest.approx(expected, rel=0.15)

    def test_a_wider_fwhm_produces_a_wider_response(self):
        """The both-ways twin for ``fwhm``.

        A knob that only reaches the config reads back correctly and blurs
        identically at every setting; this compares two settings' OUTPUTS.
        """
        unit = int(SpatialLayout(NUM_UNITS, seed=0).cell_to_unit[14, 14])
        inputs = _delta(NUM_UNITS, unit)

        narrow = GaussianReadout(fwhm=1.0, seed=0)
        wide = GaussianReadout(fwhm=4.0, seed=0)
        difference = np.abs(
            ops.convert_to_numpy(narrow(inputs, training=False))
            - ops.convert_to_numpy(wide(inputs, training=False))
        )
        assert float(difference.max()) > 1e-3

    def test_the_in_graph_blur_matches_the_numpy_reference(self):
        """Two producers of one quantity must not drift.

        ``scipy.ndimage.gaussian_filter(mode="nearest")`` replicates the border
        sample; a zero-padded ``'same'`` convolution does not. Agreement is
        therefore a measured claim, not an assumed one: max |delta| below 1e-4
        against a signal whose absmax is ~4.
        """
        layout = SpatialLayout(NUM_UNITS, seed=0)
        layer = GaussianReadout(fwhm=2.0, unit_spacing=1.0, permute=True, seed=0)
        activations = smooth_field_activations(layout, num_samples=8, seed=0)

        in_graph = ops.convert_to_numpy(
            layer(activations.astype("float32")[None, ...], training=False)
        )[0]
        reference = _numpy_blur(activations, layout, layer.sigma)

        np.testing.assert_allclose(
            in_graph, reference, atol=1e-4, rtol=0,
            err_msg="the in-graph blur and the scipy reference disagree",
        )

    def test_the_blur_follows_the_row_and_not_the_column(self):
        """The transposed-stride twin, on a non-square grid.

        Grid-adjacency along a COLUMN (same column, next row) must register. A
        stride transposed between the two spatial axes would satisfy the row test
        below and fail this one, which is the whole point of running both.
        """
        units = 512  # 16 x 32, non-square
        layout = SpatialLayout(units, seed=0)
        layer = GaussianReadout(permute=True, seed=0)

        above = int(layout.cell_to_unit[7, 10])
        below = int(layout.cell_to_unit[8, 10])
        base = _delta(units, above)
        blended = _delta(units, above, blend_with=below)

        column_delta = float(
            np.abs(
                ops.convert_to_numpy(layer(blended, training=False))
                - ops.convert_to_numpy(layer(base, training=False))
            ).max()
        )
        assert column_delta > 1e-4, (
            "column-adjacent units did not influence each other -- the two "
            "spatial axes are transposed"
        )

    def test_the_two_padding_modes_differ_at_the_border_only(self):
        """``padding_mode`` must reach the layer, not just its config.

        The two modes agree in the interior and diverge at the edge, which is
        exactly where zero-padding loses mass. An inert flag would fail the first
        assertion; a flag applied in the wrong PLACE would fail the second.
        """
        layout = SpatialLayout(NUM_UNITS, seed=0)
        inputs = _delta(NUM_UNITS, int(layout.cell_to_unit[0, 0]))

        nearest = ops.convert_to_numpy(
            GaussianReadout(padding_mode="nearest", seed=0)(inputs, training=False)
        )
        zeros = ops.convert_to_numpy(
            GaussianReadout(padding_mode="zeros", seed=0)(inputs, training=False)
        )
        assert float(np.abs(nearest - zeros).max()) > 1e-6

        grid_nearest, grid_zeros = layout.to_grid(nearest[0, 0]), layout.to_grid(
            zeros[0, 0]
        )
        np.testing.assert_allclose(
            grid_nearest[14, 14], grid_zeros[14, 14], atol=1e-6, rtol=0
        )

    def test_zeros_padding_loses_mass_at_the_border(self):
        """The defect the default avoids, pinned so the default cannot regress.

        A constant field under zero padding decays at the border; under edge
        replication it does not. Both numbers are measured rather than asserted
        from the config.
        """
        inputs = np.full((1, 1, NUM_UNITS), 1.0, dtype="float32")
        nearest = ops.convert_to_numpy(
            GaussianReadout(padding_mode="nearest", seed=0)(inputs, training=False)
        )
        zeros = ops.convert_to_numpy(
            GaussianReadout(padding_mode="zeros", seed=0)(inputs, training=False)
        )
        assert float(nearest.min()) == pytest.approx(1.0, abs=1e-4)
        assert float(zeros.min()) < 0.95

    def test_the_blur_follows_the_grid_and_not_the_unit_axis(self):
        """Orientation: a square grid cannot see a transposed stride.

        Two units that are ADJACENT ON THE GRID are far apart on the unit axis
        under a permutation, and vice versa. Smoothing the flat axis would leave
        the grid neighbours untouched; smoothing the grid leaves the flat-axis
        neighbours untouched. A 16x32 grid is used so a row/column confusion has
        somewhere to hide and be caught.
        """
        units = 512  # 16 x 32, non-square
        layout = SpatialLayout(units, seed=0)
        layer = GaussianReadout(permute=True, seed=0)

        grid_neighbour, flat_neighbour = _pick_pair(layout)
        base = _delta(units, grid_neighbour)
        blended = _delta(units, grid_neighbour, blend_with=flat_neighbour)

        grid_delta = float(
            np.abs(
                ops.convert_to_numpy(layer(blended, training=False))
                - ops.convert_to_numpy(layer(base, training=False))
            ).max()
        )
        assert grid_delta > 1e-4, (
            "grid-adjacent units did not influence each other -- the blur is "
            "running along the unit axis, not along the grid"
        )

    def test_a_permuted_and_an_identity_layout_blur_differently(self):
        """``permute`` must reach the layer, not just its config.

        Under the default permutation the grid-adjacent pair above shares nothing
        with the flat-adjacent pair, so a layout-blind blur would give identical
        outputs for both.
        """
        layout = SpatialLayout(NUM_UNITS, seed=0)
        grid_neighbour, flat_neighbour = _pick_pair(layout)
        blended = _delta(NUM_UNITS, grid_neighbour, blend_with=flat_neighbour)

        permuted = GaussianReadout(permute=True, seed=0)
        identity = GaussianReadout(permute=False)
        first = ops.convert_to_numpy(permuted(blended, training=False))
        second = ops.convert_to_numpy(identity(blended, training=False))
        assert float(np.abs(first - second).max()) > 1e-6

    def test_a_dynamic_last_axis_is_refused_at_build(self):
        layer = GaussianReadout()
        with pytest.raises(ValueError, match="statically known last axis"):
            layer.build((None, 4, None))


class TestGaussianReadoutBuild:
    def test_the_kernel_survives_the_stateless_build_pass(self):
        """Built through a parent, which is the only path that runs in the scope."""
        child = GaussianReadout(fwhm=2.0, seed=0, name="readout")
        parent = keras.Sequential([child], name="model")
        parent.build((None, 4, NUM_UNITS))

        kernel = ops.convert_to_numpy(child.blur.kernel)[:, :, 0, 0]
        assert not np.all(kernel == 0), "the kernel is all zeros"
        np.testing.assert_allclose(
            float(kernel.sum()), 1.0, atol=1e-5, rtol=0
        )

    def test_explicit_build_matches_lazy_build(self):
        def build():
            return keras.Sequential(
                [GaussianReadout(seed=0, name="readout")], name="model"
            )

        explicit = build()
        explicit.build((None, 4, NUM_UNITS))
        lazy = build()
        lazy(np.zeros((1, 4, NUM_UNITS), dtype="float32"))

        def relative(model):
            return sorted(w.path.split("/", 1)[-1] for w in model.weights)

        assert relative(explicit) == relative(lazy)


class TestGaussianReadoutPrecision:
    @pytest.mark.parametrize("policy", ("float32", "mixed_float16"))
    def test_the_output_is_finite_under_each_policy(self, policy):
        previous = keras.mixed_precision.global_policy().name
        try:
            keras.mixed_precision.set_global_policy(policy)
            layer = GaussianReadout(seed=0)
            inputs = np.random.default_rng(0).normal(
                size=(1, 4, NUM_UNITS)
            ).astype("float32")
            output = ops.convert_to_numpy(layer(inputs, training=False))
            assert np.all(np.isfinite(output))
        finally:
            keras.mixed_precision.set_global_policy(previous)

    def test_a_tf_function_trace_matches_eager(self):
        layer = GaussianReadout(seed=0)
        inputs = np.random.default_rng(1).normal(
            size=(1, 4, NUM_UNITS)
        ).astype("float32")

        eager = ops.convert_to_numpy(layer(inputs, training=False))

        @tf.function(
            input_signature=[tf.TensorSpec([None, None, NUM_UNITS], tf.float32)]
        )
        def traced(x):
            return layer(x, training=False)

        np.testing.assert_allclose(
            ops.convert_to_numpy(traced(inputs)), eager, atol=1e-5, rtol=0
        )


class TestGaussianReadoutSerialization:
    def test_the_layout_indices_survive_a_value_round_trip(self):
        """``atol=0.0``, compared before the loaded model's first call."""
        layer = GaussianReadout(seed=4, name="readout")
        model = keras.Sequential([layer], name="model")
        inputs = np.random.default_rng(0).normal(
            size=(1, 4, NUM_UNITS)
        ).astype("float32")
        model(inputs, training=False)

        saved = {
            weight.name: ops.convert_to_numpy(weight).copy()
            for weight in model.weights
        }
        assert saved

        import os
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "readout.keras")
            model.save(path)
            loaded = keras.models.load_model(path)
            restored = {
                weight.name: ops.convert_to_numpy(weight)
                for weight in loaded.weights
            }
            assert sorted(saved) == sorted(restored)
            for name, values in saved.items():
                np.testing.assert_array_equal(values, restored[name])

    def test_the_forward_output_survives_a_value_round_trip(self):
        layer = GaussianReadout(seed=0, name="readout")
        model = keras.Sequential([layer], name="model")
        inputs = np.random.default_rng(0).normal(
            size=(1, 4, NUM_UNITS)
        ).astype("float32")
        original = ops.convert_to_numpy(model(inputs, training=False))

        import os
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "readout.keras")
            model.save(path)
            loaded = keras.models.load_model(path)
            restored = ops.convert_to_numpy(loaded(inputs, training=False))

        np.testing.assert_allclose(restored, original, atol=1e-6, rtol=0)


def _pick_pair(layout):
    """A grid-adjacent pair and a flat-adjacent pair, on the same row.

    Returns ``(grid_neighbour_unit, flat_neighbour_unit)``. Both share a grid row
    and the flat neighbour is the next unit index, so a blur running along either
    axis produces a different answer and the test can tell them apart.
    """
    height, _ = layout.grid_shape
    row = height // 2
    left = int(layout.cell_to_unit[row, 4])
    right = int(layout.cell_to_unit[row, 5])
    assert left != right
    flat = int(np.flatnonzero(layout.perm == right)[0])
    return left, flat


def _delta(units, source_unit, blend_with=None):
    """A one-hot input, optionally with a second unit set to 1."""
    inputs = np.zeros((1, 1, units), dtype="float32")
    inputs[0, 0, source_unit] = 1.0
    if blend_with is not None:
        inputs[0, 0, blend_with] = 1.0
    return inputs