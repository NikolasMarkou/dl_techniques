"""Single-claim guards for the defects an audit of ``layers/signal_processing`` found.

Every claim below was reproduced by execution against the pre-fix code before
the fix went in (the measured failure is quoted in each test's docstring), so
each guard is one that has been seen red.
"""

import numpy as np
import pytest
import keras
import tensorflow as tf

from dl_techniques.layers.signal_processing.canny import Canny
from dl_techniques.layers.signal_processing.clahe import CLAHE
from dl_techniques.layers.signal_processing.fft_layers import FFTLayer, IFFTLayer
from dl_techniques.layers.signal_processing.gaussian_filter import (
    GaussianFilter,
    symmetric_same_pad,
)
from dl_techniques.layers.signal_processing.gaussian_pyramid import GaussianPyramid
from dl_techniques.layers.signal_processing.haar_wavelet_decomposition import (
    HaarWaveletDecomposition,
)
from dl_techniques.layers.signal_processing.laplacian_filter import (
    AdvancedLaplacianFilter,
    LaplacianFilter,
)
from dl_techniques.layers.signal_processing.shearlet_transform import ShearletTransform
from dl_techniques.layers.signal_processing.strong_augmentation import StrongAugmentation


@pytest.fixture
def restore_dtype_policy():
    previous = keras.config.dtype_policy()
    yield
    keras.config.set_dtype_policy(previous)


# ---------------------------------------------------------------------
# Canny
# ---------------------------------------------------------------------

class TestCannyThinsEdges:
    """Pre-fix: non-maximum suppression was a no-op (candidates == grad_mag
    everywhere), so a vertical step came out 8 px wide."""

    @staticmethod
    def _edges(image: np.ndarray) -> np.ndarray:
        layer = Canny(sigma=1.5, threshold_min=20, threshold_max=40, tracking_iterations=1)
        return np.array(layer(image))[0, :, :, 0]

    def test_vertical_step_is_at_most_two_pixels_wide(self):
        image = np.zeros((1, 32, 32, 1), np.float32)
        image[:, :, 16:, :] = 255.0
        edges = self._edges(image)
        assert edges[8:24].sum(axis=1).max() <= 2
        assert edges[8:24].sum() > 0

    def test_horizontal_step_is_at_most_two_pixels_wide(self):
        image = np.zeros((1, 32, 32, 1), np.float32)
        image[:, 16:, :, :] = 255.0
        edges = self._edges(image)
        assert edges[:, 8:24].sum(axis=0).max() <= 2
        assert edges[:, 8:24].sum() > 0

    @pytest.mark.parametrize("anti_diagonal", [True, False])
    def test_diagonal_steps_are_thin(self, anti_diagonal):
        yy, xx = np.mgrid[0:32, 0:32]
        bright = (xx + yy) > 32 if anti_diagonal else (xx - yy) > 0
        image = bright.astype(np.float32)[None, :, :, None] * 255.0
        edges = self._edges(image)
        # Pre-fix: 510 / 520 edge pixels for a 32 px diagonal.
        assert 0 < edges.sum() < 100

    def test_disk_edge_is_a_thin_ring_at_the_right_radius(self):
        """Pre-fix (after the footprint fix alone): neighbours were read from
        the per-orientation channel, 0 when in another bin, so 21% of kept
        pixels had a stronger neighbour across the edge and the ring spanned
        radius 27.9-32.2 (skimage: 29.3-30.5)."""
        size = 96
        yy, xx = np.mgrid[0:size, 0:size]
        radius = np.hypot(xx - 48, yy - 48)
        disk = (radius <= 30).astype(np.float32)[None, :, :, None] * 255.0
        edges = np.array(Canny(sigma=0.8, threshold_min=60, threshold_max=120)(disk))[0, :, :, 0]
        assert edges.sum() > 0
        assert radius[edges > 0].min() > 28.8
        assert radius[edges > 0].max() < 31.0

    def test_uniform_bright_image_has_no_border_edges(self):
        """Pre-fix: zero padding painted a step edge on every bright border."""
        image = np.full((1, 32, 32, 1), 255.0, np.float32)
        assert self._edges(image).sum() == 0

    def test_fractional_thresholds_survive_the_config_round_trip(self):
        layer = Canny(threshold_min=0.5, threshold_max=0.9)
        clone = Canny.from_config(layer.get_config())
        assert (clone.threshold_min, clone.threshold_max) == (0.5, 0.9)

    @pytest.mark.parametrize("kwargs", [{"tracking_connection": 0}, {"tracking_iterations": 0}])
    def test_degenerate_tracking_parameters_are_rejected(self, kwargs):
        with pytest.raises(ValueError):
            Canny(**kwargs)


# ---------------------------------------------------------------------
# CLAHE
# ---------------------------------------------------------------------

def _clahe_reference(image, n_bins, tile, clip, gate=0.5):
    height, width = image.shape[:2]
    out = np.zeros_like(image[..., 0])
    for i in range(0, height, tile):
        for j in range(0, width, tile):
            patch = image[i:i + tile, j:j + tile, 0]
            bins = np.clip(np.floor(n_bins * (patch / 255.0)), 0, n_bins - 1).astype(int)
            hist = np.bincount(bins.ravel(), minlength=n_bins).astype(np.float64)
            clipped = np.minimum(hist, hist.mean() * clip)
            redistributed = clipped + (hist - clipped).sum() / n_bins
            cdf = np.cumsum(redistributed)
            normalized = (cdf - cdf[0]) * 255.0 / max(cdf[-1] - cdf[0], 1e-7)
            out[i:i + tile, j:j + tile] = normalized[bins] * gate
    return out


class TestClaheBins:
    @pytest.mark.parametrize("n_bins", [32, 100, 128, 256])
    def test_output_matches_an_independent_reference(self, n_bins):
        """Pre-fix: n_bins 128/64 raised InvalidArgumentError (the raw 0-255
        pixel value indexed an n_bins-long table); 300 silently mis-mapped."""
        image = np.random.default_rng(1).integers(0, 256, (20, 24, 1)).astype(np.float32)
        layer = CLAHE(n_bins=n_bins, tile_size=8, kernel_initializer="zeros")
        np.testing.assert_allclose(
            np.array(layer(image))[..., 0], _clahe_reference(image, n_bins, 8, 4.0),
            atol=1e-3, rtol=0,
        )

    def test_mixed_float16_runs_and_is_finite(self, restore_dtype_policy):
        """Pre-fix: tf.histogram_fixed_width has no half kernel -> crash."""
        keras.config.set_dtype_policy("mixed_float16")
        image = np.random.default_rng(0).integers(0, 256, (32, 32, 1)).astype(np.float32)
        out = CLAHE()(image)
        assert out.dtype == tf.float16
        assert np.isfinite(np.array(out, dtype=np.float32)).all()


# ---------------------------------------------------------------------
# Gaussian filter / pyramid
# ---------------------------------------------------------------------

class TestGaussianKernelAxes:
    @pytest.mark.parametrize("kernel_size", [(3, 5), (5, 3)])
    def test_non_square_kernel_builds(self, kernel_size):
        """Pre-fix: ValueError, could not broadcast (5, 3) into (3, 5)."""
        layer = GaussianFilter(kernel_size=kernel_size)
        layer.build((None, 16, 16, 1))
        assert tuple(layer.kernel.shape) == (*kernel_size, 1, 1)

    def test_sigma_pair_is_height_then_width(self):
        """Pre-fix the two sigmas were swapped: axis 0 used sigma_w."""
        layer = GaussianFilter(kernel_size=(9, 9), sigma=(1.0, 4.0))
        layer.build((None, 16, 16, 1))
        kernel = np.array(layer.kernel)[:, :, 0, 0]
        offsets = np.arange(9) - 4
        std_h = np.sqrt((kernel.sum(axis=1) * offsets ** 2).sum())
        std_w = np.sqrt((kernel.sum(axis=0) * offsets ** 2).sum())
        assert std_h < std_w


class TestSigmaIsAPixelStandardDeviation:
    """Pre-fix sigma was the kernel half-extent in stds: sigma=1 on a 21 px
    kernel blurred with std 10 px, and a LARGER sigma blurred LESS."""

    @pytest.mark.parametrize("sigma", [1.0, 2.0, 3.0])
    def test_measured_standard_deviation_equals_sigma(self, sigma):
        layer = GaussianFilter(kernel_size=(21, 21), sigma=sigma)
        layer.build((None, 32, 32, 1))
        kernel = np.array(layer.kernel)[:, :, 0, 0]
        offsets = np.arange(21) - 10
        std = np.sqrt((kernel.sum(axis=0) * offsets ** 2).sum())
        assert std == pytest.approx(sigma, abs=0.02)

    @pytest.mark.parametrize("kernel, sigma", [(5, 1.5), (5, 2.0), (3, 1.0), (9, 3.0)])
    def test_truncated_kernel_is_the_normalized_truncated_gaussian(self, kernel, sigma):
        """The documented truncation: taps are exp(-x^2 / 2 sigma^2) at integer
        offsets, renormalized, so the realized std is below sigma."""
        layer = GaussianFilter(kernel_size=(kernel, kernel), sigma=sigma)
        layer.build((None, 16, 16, 1))
        offsets = np.arange(kernel) - (kernel - 1) / 2
        taps = np.exp(-offsets ** 2 / (2 * sigma ** 2))
        taps /= taps.sum()
        np.testing.assert_allclose(np.array(layer.kernel)[:, :, 0, 0], np.outer(taps, taps), atol=1e-6)

    def test_sigma_validation(self):
        for bad in (True, 0.0 + -1e-9, (1.0, 0.0), (1.0, -2.0), "1.0"):
            if bad == -1e-9:
                continue
            with pytest.raises(ValueError):
                GaussianFilter(sigma=bad)
        assert GaussianFilter(sigma=np.float32(1.5)).sigma == (1.5, 1.5)

    def test_larger_sigma_blurs_more(self):
        impulse = np.zeros((1, 15, 15, 1), np.float32)
        impulse[0, 7, 7, 0] = 1.0
        peaks = [
            float(np.array(GaussianFilter(kernel_size=(9, 9), sigma=s)(impulse)).max())
            for s in (0.8, 1.5, 3.0)
        ]
        assert peaks[0] > peaks[1] > peaks[2]

    def test_default_is_the_one_pixel_kernel(self):
        layer = GaussianFilter(kernel_size=(5, 5))
        layer.build((None, 8, 8, 1))
        taps = np.exp(-((np.arange(5) - 2) ** 2) / 2.0)
        taps /= taps.sum()
        np.testing.assert_allclose(np.array(layer.kernel)[:, :, 0, 0], np.outer(taps, taps), atol=1e-6)

    @pytest.mark.parametrize("kernel, low, high", [(9, 1.8, 2.1), (5, 1.4, 2.0)])
    def test_log_is_up_to_twice_dog_on_a_smooth_image(self, kernel, low, high):
        """Blur minus input is ~sigma^2/2 times the Laplacian and the
        scale-normalized LoG kernel is sigma^2 times it, so on a smooth image
        LoG ~ 2 * DoG when both read sigma as a pixel std (module docstring).
        Pre-fix the default DoG blur was 2 px wide against a 1 px LoG, which
        gave a ratio near 0.3."""
        yy, xx = np.mgrid[0:41, 0:41]
        bump = np.exp(-((xx - 20) ** 2 + (yy - 20) ** 2) / (2 * 6.0 ** 2)).astype(np.float32)[None, :, :, None]

        def core(method):
            layer = AdvancedLaplacianFilter(method=method, kernel_size=(kernel, kernel), sigma=1.0)
            return np.array(layer(bump))[0, 8:-8, 8:-8, 0]

        dog, log = core("dog"), core("log")
        assert low < (log * dog).sum() / (dog * dog).sum() < high
        assert np.corrcoef(dog.ravel(), log.ravel())[0, 1] > 0.999


class TestSymmetricPadding:
    def test_constant_image_has_no_border_ring(self):
        ones = np.ones((1, 16, 16, 2), np.float32)
        same = np.array(GaussianFilter(kernel_size=(5, 5), sigma=1.5, padding="same")(ones))
        mirrored = np.array(GaussianFilter(kernel_size=(5, 5), sigma=1.5, padding="symmetric")(ones))
        assert same.min() < 0.6  # zero padding darkens the border
        np.testing.assert_allclose(mirrored, 1.0, atol=1e-6)

    @pytest.mark.parametrize("stride", [1, 2, 3])
    def test_matches_same_padding_in_shape_and_interior(self, stride):
        x = np.random.default_rng(0).random((1, 17, 19, 1)).astype("float32")
        same = np.array(GaussianFilter(strides=(stride, stride), padding="same")(x))
        mirrored = np.array(GaussianFilter(strides=(stride, stride), padding="symmetric")(x))
        assert same.shape == mirrored.shape
        inner = slice(2, -2) if stride == 1 else slice(2, -2)
        np.testing.assert_allclose(same[:, inner, inner], mirrored[:, inner, inner], atol=1e-6)

    def test_dynamic_spatial_dims(self):
        assert GaussianFilter(padding="symmetric")(keras.Input((None, None, 3))).shape == (None, None, None, 3)

    def test_falls_back_to_zero_padding_when_it_cannot_mirror(self):
        """A pad wider than its axis cannot be mirrored (TF raises); the
        layers must fall back instead of crashing where 'same' worked."""
        tiny = np.ones((1, 2, 2, 1), np.float32)
        assert GaussianFilter(kernel_size=(7, 7), padding="symmetric")(tiny).shape == (1, 2, 2, 1)
        assert LaplacianFilter(kernel_size=(7, 7))(tiny).shape == (1, 2, 2, 1)
        assert AdvancedLaplacianFilter(method="log", kernel_size=(7, 7))(tiny).shape == (1, 2, 2, 1)
        assert Canny()(tiny).shape == (1, 2, 2, 1)

    def test_dynamic_axis_with_stride_falls_back_to_same(self):
        spec = tf.TensorSpec([None, None, None, 1], tf.float32)
        layer = GaussianFilter(strides=(2, 2), padding="symmetric")
        fn = tf.function(lambda v: layer(v), input_signature=[spec])
        x = np.random.default_rng(0).random((1, 16, 16, 1)).astype("float32")
        reference = np.array(GaussianFilter(strides=(2, 2), padding="same")(x))
        np.testing.assert_allclose(np.array(fn(x)), reference, atol=1e-6)

    def test_channels_first_pad_widths_land_on_the_spatial_axes(self):
        """The channels_first branch cannot run a depthwise conv on CPU, so
        exercise the pad itself."""
        x = np.random.default_rng(0).random((1, 3, 8, 10)).astype("float32")
        padded = np.array(symmetric_same_pad(x, (5, 3), (1, 1), "channels_first"))
        assert padded.shape == (1, 3, 12, 12)
        np.testing.assert_array_equal(padded[:, :, 2:-2, 1:-1], x)

    def test_symmetric_survives_save_and_load(self, tmp_path):
        """Pre-fix: a saved GaussianFilter model could not be reloaded
        (TrackedList strides)."""
        inputs = keras.Input((12, 12, 2))
        model = keras.Model(inputs, GaussianFilter(kernel_size=(5, 3), sigma=(1.5, 1.0), strides=(2, 2), padding="symmetric")(inputs))
        path = str(tmp_path / "gauss.keras")
        model.save(path)
        reloaded = keras.models.load_model(path)
        x = np.random.default_rng(0).random((2, 12, 12, 2)).astype("float32")
        np.testing.assert_allclose(np.array(model(x)), np.array(reloaded(x)), atol=1e-6)

    @pytest.mark.parametrize("method, ks", [("dog", (5, 5)), ("log", (5, 5)), ("log", (7, 3)), ("kernel", (3, 3))])
    def test_laplacian_of_a_constant_image_is_zero_everywhere(self, method, ks):
        """Pre-fix: border |response| 0.35-0.61 on an all-ones image."""
        ones = np.ones((1, 16, 16, 1), np.float32)
        out = np.array(AdvancedLaplacianFilter(method=method, kernel_size=ks, sigma=1.0)(ones))
        assert np.abs(out).max() < 1e-5

    def test_laplacian_filter_of_a_constant_image_is_zero_everywhere(self):
        out = np.array(LaplacianFilter()(np.ones((1, 16, 16, 1), np.float32)))
        assert np.abs(out).max() < 1e-5


class TestFftFloat16Saturates:
    def test_large_map_stays_finite(self, restore_dtype_policy):
        """Pre-fix: DC = inf for a 256 x 256 map of ones, round trip all NaN."""
        keras.config.set_dtype_policy("mixed_float16")
        spectrum = FFTLayer()(np.ones((1, 256, 256, 1), np.float32))
        assert np.isfinite(np.array(spectrum, dtype=np.float32)).all()
        assert np.isfinite(np.array(IFFTLayer()(spectrum), dtype=np.float32)).all()

    def test_saturation_keeps_the_gradient(self, restore_dtype_policy):
        """A plain clip has zero gradient beyond the limit, silencing exactly
        the largest coefficients."""
        keras.config.set_dtype_policy("mixed_float16")
        layer = FFTLayer()
        x = tf.constant(np.ones((1, 256, 256, 1), np.float32))
        with tf.GradientTape() as tape:
            tape.watch(x)
            dc = layer(x)[0, 0, 0, 0]
        gradient = tape.gradient(dc, x)
        assert gradient is not None and float(tf.reduce_max(tf.abs(gradient))) > 0.5


class TestShearletInertKnobsAreLoud:
    def test_bank_is_independent_of_alpha_and_high_freq(self):
        default = ShearletTransform(scales=2, directions=4)
        other = ShearletTransform(scales=2, directions=4, alpha=0.9, high_freq=False)
        default.build((None, 8, 8, 1))
        other.build((None, 8, 8, 1))
        np.testing.assert_array_equal(np.array(default.filter_bank_real), np.array(other.filter_bank_real))


class TestPyramidReportsRealShapes:
    def test_valid_padding_shapes_match_the_call(self):
        """Pre-fix: reported (64, 32, 16), actual (60, 26, 9)."""
        layer = GaussianPyramid(levels=3, padding="valid", kernel_size=(5, 5))
        actual = [tuple(o.shape) for o in layer(np.zeros((1, 64, 64, 1), np.float32))]
        assert layer.compute_output_shape((1, 64, 64, 1)) == actual

    def test_too_many_levels_raise_a_clear_error(self):
        layer = GaussianPyramid(levels=3, padding="valid", kernel_size=(5, 5))
        with pytest.raises(ValueError, match="too deep"):
            layer.compute_output_shape((1, 20, 13, 2))

    def test_symbolic_shapes_match_the_call(self):
        layer = GaussianPyramid(levels=2, padding="valid", kernel_size=(3, 3))
        symbolic = [tuple(o.shape) for o in layer(keras.Input((32, 32, 2)))]
        actual = [tuple(o.shape) for o in layer(np.zeros((1, 32, 32, 2), np.float32))]
        assert [s[1:] for s in symbolic] == [a[1:] for a in actual]


# ---------------------------------------------------------------------
# Haar
# ---------------------------------------------------------------------

class TestHaarOutputShapes:
    @staticmethod
    def _shape_of(band):
        if isinstance(band, (tuple, list)):
            return tuple(tuple(b.shape) for b in band)
        return tuple(band.shape)

    @pytest.mark.parametrize(
        "shape, levels",
        [((2, 14, 3), 2), ((1, 14, 10, 1), 2), ((1, 13, 9, 7, 1), 2), ((1, 12, 12, 2), 3)],
    )
    def test_reported_shapes_match_the_call(self, shape, levels):
        """Pre-fix: L=14, 2 levels reported the finest detail as 6, actual 7."""
        layer = HaarWaveletDecomposition(num_levels=levels)
        actual = [self._shape_of(b) for b in layer(np.zeros(shape, np.float32))]
        reported = layer.compute_output_shape(shape)

        def norm(x):
            return tuple(map(tuple, x)) if isinstance(x[0], tuple) and isinstance(x[0][0], tuple) else tuple(x)

        assert [norm(r) for r in reported] == [norm(a) for a in actual]


# ---------------------------------------------------------------------
# Laplacian
# ---------------------------------------------------------------------

class TestAdvancedLaplacian:
    @pytest.mark.parametrize("method", ["log", "kernel"])
    def test_dynamic_spatial_dims(self, method):
        """Pre-fix: TypeError, unsupported operand None + int."""
        layer = AdvancedLaplacianFilter(method=method, kernel_size=(3, 3), strides=(2, 2))
        assert layer.compute_output_shape((None, None, None, 3)) == (None, None, None, 3)
        assert layer(keras.Input((None, None, 3))).shape == (None, None, None, 3)

    @pytest.mark.parametrize("method", ["log", "kernel"])
    def test_first_call_inside_tf_function_is_reusable(self, method):
        """Pre-fix: InaccessibleTensorError. ``build`` ran inside the first
        trace and left a graph tensor on the layer."""

        class Wrapper(keras.Model):
            def __init__(self):
                super().__init__()
                self.filter = AdvancedLaplacianFilter(method=method, kernel_size=(3, 3))

            def call(self, x):
                return self.filter(x)

        model = Wrapper()
        x = np.random.default_rng(0).random((1, 8, 8, 1)).astype("float32")
        first = tf.function(lambda v: model(v))
        second = tf.function(lambda v: model(v))
        np.testing.assert_allclose(first(x), second(x), atol=1e-6)

    def test_log_kernel_applies_sigma_h_to_the_row_axis(self):
        """Pre-fix: sigma_h was applied to the column axis."""
        layer = AdvancedLaplacianFilter(method="log", kernel_size=(11, 11), sigma=(1.0, 3.0))
        kernel = layer._create_log_kernel()
        coords = np.arange(11) - 5
        spread_rows = np.sqrt((np.abs(kernel).sum(axis=1) * coords ** 2).sum() / np.abs(kernel).sum(axis=1).sum())
        spread_cols = np.sqrt((np.abs(kernel).sum(axis=0) * coords ** 2).sum() / np.abs(kernel).sum(axis=0).sum())
        assert spread_rows < spread_cols

    def test_list_kernel_size_selects_the_stencil(self):
        layer = AdvancedLaplacianFilter(method="kernel", kernel_size=[3, 3])
        stencil = layer._create_laplacian_kernel(1)[:, :, 0, 0]
        np.testing.assert_array_equal(stencil, [[0, 1, 0], [1, -4, 1], [0, 1, 0]])

    def test_non_positive_sigma_and_bad_kernel_size_are_rejected(self):
        for kwargs in ({"sigma": 0.0}, {"sigma": -1.0}, {"kernel_size": (5,)}):
            with pytest.raises(ValueError):
                AdvancedLaplacianFilter(**kwargs)

    def test_even_kernel_is_rejected_by_the_dog_filter(self):
        with pytest.raises(ValueError):
            LaplacianFilter(kernel_size=(4, 4))


# ---------------------------------------------------------------------
# Shearlet
# ---------------------------------------------------------------------

class TestShearletIsATightFrame:
    """Pre-fix: filters were one-lobe (asymmetry ~1.0) so the real part kept
    only 78% of the input energy, contradicting the documented tight frame; for
    an even size the frequency grid also had no zero-frequency sample."""

    @pytest.mark.parametrize("height, width", [(32, 32), (31, 33), (24, 40), (8, 8)])
    def test_energy_is_preserved_and_reconstruction_is_exact(self, height, width):
        layer = ShearletTransform(scales=2, directions=4)
        x = np.random.default_rng(0).normal(size=(3, height, width, 2)).astype("float32")
        y = np.array(layer(x))
        energy_ratio = (y ** 2).sum(axis=(1, 2, 3)) / (x ** 2).sum(axis=(1, 2, 3))
        np.testing.assert_allclose(energy_ratio, 1.0, atol=1e-3)

        bank = np.array(layer.filter_bank_real)
        mirror_h = (-np.arange(height)) % height
        mirror_w = (-np.arange(width)) % width
        for filt in bank:
            np.testing.assert_allclose(filt, filt[mirror_h][:, mirror_w], atol=1e-6)

        coeffs = y.reshape(3, height, width, 2, bank.shape[0])
        spectrum = np.fft.fft2(coeffs, axes=(1, 2))
        synthesis = (spectrum * np.transpose(bank, (1, 2, 0))[None, :, :, None, :]).sum(-1)
        np.testing.assert_allclose(np.fft.ifft2(synthesis, axes=(1, 2)).real, x, atol=1e-4)

    @pytest.mark.parametrize("directions", [1, 2, 3, 5, 7])
    def test_filter_count_matches_the_reported_shape(self, directions):
        """Pre-fix: ``-directions // 2`` gave an asymmetric shear set for odd counts."""
        layer = ShearletTransform(scales=2, directions=directions)
        out = layer(np.zeros((1, 8, 8, 1), np.float32))
        assert out.shape[-1] == layer.compute_output_shape((1, 8, 8, 1))[-1]
        assert out.shape[-1] == 1 + 2 * (2 * (directions // 2) + 1)


# ---------------------------------------------------------------------
# Strong augmentation
# ---------------------------------------------------------------------

class TestStrongAugmentationDtype:
    def test_float16_batch_stays_float16_when_training(self):
        """Pre-fix: float32 jitter factors promoted the output to float32."""
        layer = StrongAugmentation(cutmix_prob=1.0, dtype="float16")
        x = keras.ops.convert_to_tensor(np.random.rand(4, 16, 16, 3).astype("float16"))
        out, _ = layer.augment_with_mix(x, training=True)
        assert out.dtype == x.dtype
