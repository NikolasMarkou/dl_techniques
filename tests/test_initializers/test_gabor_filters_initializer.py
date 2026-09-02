"""
Comprehensive test suite for GaborFiltersInitializer.

This module contains test cases for the GaborFiltersInitializer, ensuring
correct construction / validation, Gabor filter-bank distribution,
normalization, determinism, dtype handling, serialization round-trips, and
integration with Keras layers (Conv2D save/load).

Several tests here are single-claim guards for defects found by review and
verified by measurement against the previous implementation:

* an unvalidated ``lambda_range`` whose minimum was 0 produced an all-NaN
  kernel (25/25 NaN on a ``(5, 5, 1, 1)`` shape) with no exception,
* a non-finite range bound passed both comparisons (``nan > hi`` and
  ``nan <= 0`` are both False) and did the same,
* the bank was neither DC-removed nor energy-normalized: on ``(11, 11, 3, 96)``
  per-filter L2 norms spanned 0.12..4.60 and per-output-channel ``sum |w|``
  spanned 0.54..100.3,
* the joint-``linspace`` sweep made all five parameters rank-1 correlated: max
  off-diagonal cosine similarity 0.9999, 29 pairs above 0.95.
"""

import os
import tempfile
from typing import Tuple
from unittest import mock

import keras
import numpy as np
import pytest

from dl_techniques.initializers import (
    GaborFiltersInitializer,
    create_gabor_conv2d,
    create_gabor_depthwise_conv2d,
)
from dl_techniques.initializers.gabor_filters_initializer import (
    LAMBDA_FRACTIONS,
    SIGMA_FRACTIONS,
    logger as gabor_logger,
)


def _cosine_gram(bank_2d: np.ndarray) -> np.ndarray:
    """Off-diagonal cosine-similarity matrix of a ``(kh, kw, n)`` filter bank."""
    flat = bank_2d.reshape(-1, bank_2d.shape[-1])
    flat = flat / (np.linalg.norm(flat, axis=0, keepdims=True) + 1e-12)
    gram = flat.T @ flat
    np.fill_diagonal(gram, 0.0)
    return gram


class TestGaborFiltersInitializer:
    """Test suite for GaborFiltersInitializer implementation."""

    @pytest.fixture
    def standard_shape(self) -> Tuple[int, int, int, int]:
        """Standard 4D Conv2D kernel shape (kh, kw, in_ch, out_ch)."""
        return (5, 5, 3, 96)

    @pytest.fixture
    def initializer(self) -> GaborFiltersInitializer:
        """Default initializer instance."""
        return GaborFiltersInitializer()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def test_initialization_defaults(self):
        """Defaults: kernel-relative scales, half-open periodic ranges, product sweep."""
        init = GaborFiltersInitializer()

        # sigma and lambda are resolved against the kernel size at call time.
        assert init.sigma_range is None
        assert init.lambda_range is None

        assert init.theta_range == (0.0, 180.0)
        assert init.gamma_range == (0.5, 1.5)
        assert init.psi_range == (0.0, 360.0)
        assert init.sweep == "product"
        assert init.n_filters is None
        assert init.normalize is True

        for attr in (init.theta_range, init.gamma_range, init.psi_range):
            assert isinstance(attr, tuple)

    def test_initialization_custom(self):
        """Custom ranges are stored verbatim as tuples (list input coerced)."""
        init = GaborFiltersInitializer(
            sigma_range=(1.0, 10.0),
            theta_range=[0.0, 180.0],   # list input -> coerced to tuple
            lambda_range=(4.0, 50.0),
            gamma_range=(0.5, 2.0),
            psi_range=(0.0, 90.0),
        )

        assert init.sigma_range == (1.0, 10.0)
        assert init.theta_range == (0.0, 180.0)
        assert isinstance(init.theta_range, tuple)
        assert init.lambda_range == (4.0, 50.0)
        assert init.gamma_range == (0.5, 2.0)
        assert init.psi_range == (0.0, 90.0)

    def test_scale_ranges_resolve_against_the_kernel_size(self):
        """None sigma/lambda become kernel-relative fractions of min(kh, kw)."""
        init = GaborFiltersInitializer()

        for k in (5, 11, 21):
            resolved = init._resolved_ranges(k)
            assert resolved["sigma"] == pytest.approx(
                (SIGMA_FRACTIONS[0] * k, SIGMA_FRACTIONS[1] * k)
            )
            assert resolved["lambda"] == pytest.approx(
                (LAMBDA_FRACTIONS[0] * k, LAMBDA_FRACTIONS[1] * k)
            )

        # An explicit range is used as given, independent of the kernel size.
        explicit = GaborFiltersInitializer(sigma_range=(2.0, 3.0))
        assert explicit._resolved_ranges(11)["sigma"] == (2.0, 3.0)

    # ------------------------------------------------------------------
    # Parameter validation
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("kwargs", [
        {"sigma_range": (0.0, 21.0)},     # sigma min <= 0
        {"sigma_range": (-1.0, 5.0)},     # sigma min < 0
        {"theta_range": (10.0, 5.0)},     # min > max
        {"lambda_range": (8.0,)},         # not 2-element
    ])
    def test_invalid_ranges(self, kwargs):
        """Invalid construction args raise ValueError."""
        with pytest.raises(ValueError):
            GaborFiltersInitializer(**kwargs)

    @pytest.mark.parametrize("lambda_range", [(0.0, 100.0), (-1.0, 100.0)])
    def test_non_positive_lambda_is_rejected(self, lambda_range):
        """lambda divides 2*pi*x_theta, so lambda_range[0] <= 0 must raise.

        Guard for a measured defect: lambda_range=(0.0, 100.0) previously
        produced a silent all-NaN kernel (25/25 NaN on (5, 5, 1, 1)).
        """
        with pytest.raises(ValueError, match="lambda_range"):
            GaborFiltersInitializer(lambda_range=lambda_range)

    def test_negative_gamma_is_rejected(self):
        """gamma is an aspect ratio; a negative minimum silently behaved as |gamma|."""
        with pytest.raises(ValueError, match="gamma_range"):
            GaborFiltersInitializer(gamma_range=(-5.0, -1.0))

    @pytest.mark.parametrize("bad", [
        {"sigma_range": (float("nan"), 1.0)},
        {"sigma_range": (1.0, float("inf"))},
        {"theta_range": (0.0, float("nan"))},
        {"gamma_range": (float("-inf"), 1.0)},
    ])
    def test_non_finite_bounds_are_rejected(self, bad):
        """A non-finite bound passes every naive comparison, so it is rejected up front."""
        with pytest.raises(ValueError, match="finite"):
            GaborFiltersInitializer(**bad)

    def test_unknown_keyword_is_a_closed_signature(self):
        """The signature takes no **kwargs: an unknown key raises TypeError.

        keras.initializers.Initializer defines no __init__, so a **kwargs
        passthrough could only ever raise an opaque
        "object.__init__() takes exactly one argument".
        """
        with pytest.raises(TypeError):
            GaborFiltersInitializer(not_a_parameter=1)

    @pytest.mark.parametrize("kwargs", [
        {"sweep": "linear"},
        {"n_filters": 0},
        {"n_filters": -3},
    ])
    def test_invalid_sweep_and_n_filters(self, kwargs):
        """sweep must be a known mode and n_filters must be >= 1 or None."""
        with pytest.raises(ValueError):
            GaborFiltersInitializer(**kwargs)

    def test_construction_does_not_log_at_info(self):
        """Construction is silent at INFO; it happens on every deserialization."""
        with mock.patch.object(gabor_logger, "info") as info:
            GaborFiltersInitializer()
            create_gabor_depthwise_conv2d(filters_per_channel=4)
        info.assert_not_called()

    # ------------------------------------------------------------------
    # __call__ shape + finiteness
    # ------------------------------------------------------------------

    def test_call_shape(self, initializer, standard_shape):
        """__call__ returns exactly the requested 4D shape, no NaN/Inf."""
        weights = initializer(standard_shape)

        assert tuple(weights.shape) == standard_shape

        arr = np.asarray(weights)
        assert not np.any(np.isnan(arr))
        assert not np.any(np.isinf(arr))

    @pytest.mark.parametrize("shape", [
        (1, 1, 3, 4), (2, 2, 1, 4), (3, 3, 3, 8), (4, 4, 2, 4),
        (5, 5, 3, 96), (7, 7, 3, 8), (11, 11, 1, 1), (11, 11, 3, 96),
    ])
    def test_every_shape_is_finite(self, shape):
        """No shape, odd or even, tiny or large, produces a non-finite weight."""
        assert np.all(np.isfinite(np.asarray(GaborFiltersInitializer()(shape))))

    @pytest.mark.parametrize("invalid_shape", [
        (5, 5, 3),        # 3D
        (5, 5, 3, 4, 1),  # 5D
        (0, 5, 3, 4),     # zero dim
        (5, 5, 0, 4),     # zero in_ch
        (5, 5, 3, 0),     # zero out_ch
    ])
    def test_invalid_shapes(self, initializer, invalid_shape):
        """Invalid shapes raise ValueError."""
        with pytest.raises(ValueError):
            initializer(invalid_shape)

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def test_normalization_equalizes_energy_and_removes_dc(self):
        """Every filter is zero-mean with identical L2 norm and He-like RMS."""
        kh, kw, in_ch, out_ch = 11, 11, 3, 96
        weights = np.asarray(
            GaborFiltersInitializer()((kh, kw, in_ch, out_ch))
        )
        bank = weights[:, :, 0, :]

        dc = np.abs(bank.mean(axis=(0, 1)))
        assert dc.max() < 1e-6, f"DC component not removed, max |mean| = {dc.max()}"

        norms = np.sqrt((bank ** 2).sum(axis=(0, 1)))
        assert norms.max() / norms.min() == pytest.approx(1.0, abs=1e-5)

        fan_in = kh * kw * in_ch
        assert weights.std() == pytest.approx(np.sqrt(2.0 / fan_in), rel=1e-4)

    def test_normalize_false_leaves_the_raw_bank(self):
        """normalize=False keeps the un-normalized, DC-carrying Gabor responses."""
        raw = np.asarray(
            GaborFiltersInitializer(normalize=False)((11, 11, 3, 96))
        )[:, :, 0, :]

        assert np.abs(raw.mean(axis=(0, 1))).max() > 1e-3
        assert np.sqrt((raw ** 2).sum(axis=(0, 1))).min() > 1.0  # not the He scale

    def test_normalization_rescues_the_legacy_ranges(self):
        """Normalization is what fixes the measured 38x energy spread.

        The old defaults, un-normalized, spanned a factor of 38 in per-filter L2
        norm and two orders of magnitude in per-output-channel gain. The same
        parameters with normalization on are exactly equal-energy, which is the
        property that makes the bank usable as an initializer.
        """
        legacy = dict(
            sweep="diagonal",
            sigma_range=(2.0, 21.0),
            theta_range=(0.0, 360.0),
            lambda_range=(8.0, 100.0),
            gamma_range=(0.0, 300.0),
            psi_range=(0.0, 360.0),
        )
        shape = (11, 11, 3, 96)

        raw = np.asarray(
            GaborFiltersInitializer(normalize=False, **legacy)(shape)
        )
        raw_norms = np.sqrt((raw[:, :, 0, :] ** 2).sum(axis=(0, 1)))
        raw_gain = np.abs(raw).sum(axis=(0, 1, 2))
        assert raw_norms.max() / raw_norms.min() > 30.0
        assert raw_gain.max() / raw_gain.min() > 100.0

        fixed = np.asarray(GaborFiltersInitializer(**legacy)(shape))
        fixed_norms = np.sqrt((fixed[:, :, 0, :] ** 2).sum(axis=(0, 1)))
        assert fixed_norms.max() / fixed_norms.min() == pytest.approx(1.0, abs=1e-5)

    def test_a_one_by_one_kernel_is_not_dead(self):
        """DC removal is skipped at kh*kw == 1, which would otherwise zero the layer."""
        weights = np.asarray(GaborFiltersInitializer()((1, 1, 3, 4)))
        assert np.abs(weights).max() > 0.0

    # ------------------------------------------------------------------
    # Sweep quality
    # ------------------------------------------------------------------

    def test_the_product_sweep_has_no_near_duplicate_filters(self):
        """No two filters are near-collinear in the same direction.

        Anti-vacuity arm: the legacy "diagonal" sweep, which this guard exists to
        replace, DOES breach the same threshold, so the predicate discriminates.
        """
        product = np.asarray(GaborFiltersInitializer()((11, 11, 1, 96)))[:, :, 0, :]
        assert _cosine_gram(product).max() < 0.95

        diagonal = np.asarray(
            GaborFiltersInitializer(
                sweep="diagonal",
                sigma_range=(2.0, 21.0),
                theta_range=(0.0, 360.0),
                lambda_range=(8.0, 100.0),
                gamma_range=(0.0, 3.0),
                psi_range=(0.0, 360.0),
            )((11, 11, 1, 96))
        )[:, :, 0, :]
        assert _cosine_gram(diagonal).max() > 0.99

    def test_the_product_sweep_carries_phase_reversed_siblings(self):
        """Every filter has a -1 cosine sibling, so a ReLU keeps its negative lobe."""
        bank = np.asarray(GaborFiltersInitializer()((11, 11, 1, 96)))[:, :, 0, :]
        gram = _cosine_gram(bank)
        assert np.all(gram.min(axis=1) < -0.999)

    def test_the_effective_rank_beats_the_diagonal_sweep(self):
        """The factorized bank spans measurably more directions than the diagonal one."""
        def effective_rank(bank: np.ndarray) -> int:
            singular = np.linalg.svd(bank.reshape(-1, bank.shape[-1]),
                                     compute_uv=False)
            energy = np.cumsum(singular ** 2) / np.sum(singular ** 2)
            return int(np.searchsorted(energy, 0.99) + 1)

        product = np.asarray(GaborFiltersInitializer()((11, 11, 1, 96)))[:, :, 0, :]
        diagonal = np.asarray(
            GaborFiltersInitializer(
                sweep="diagonal",
                sigma_range=(2.0, 21.0),
                theta_range=(0.0, 360.0),
                lambda_range=(8.0, 100.0),
                gamma_range=(0.0, 3.0),
                psi_range=(0.0, 360.0),
            )((11, 11, 1, 96))
        )[:, :, 0, :]

        assert effective_rank(product) > effective_rank(diagonal)

    def test_the_diagonal_sweep_is_the_joint_linspace(self):
        """sweep='diagonal' reproduces the original construction exactly."""
        kh = kw = 9
        n = 6
        ranges = dict(
            sigma_range=(2.0, 21.0),
            theta_range=(0.0, 360.0),
            lambda_range=(8.0, 100.0),
            gamma_range=(0.0, 3.0),
            psi_range=(0.0, 360.0),
        )
        actual = np.asarray(
            GaborFiltersInitializer(sweep="diagonal", normalize=False, **ranges)(
                (kh, kw, 1, n)
            )
        )[:, :, 0, :]

        sigmas = np.linspace(*ranges["sigma_range"], n)
        thetas = np.deg2rad(np.linspace(*ranges["theta_range"], n))
        lambdas = np.linspace(*ranges["lambda_range"], n)
        gammas = np.linspace(*ranges["gamma_range"], n)
        psis = np.deg2rad(np.linspace(*ranges["psi_range"], n))

        xs = np.arange(kw) - (kw - 1) / 2.0
        ys = np.arange(kh) - (kh - 1) / 2.0
        xx, yy = np.meshgrid(xs, ys)

        for j in range(n):
            x_t = xx * np.cos(thetas[j]) + yy * np.sin(thetas[j])
            y_t = -xx * np.sin(thetas[j]) + yy * np.cos(thetas[j])
            expected = np.exp(
                -(x_t ** 2 + gammas[j] ** 2 * y_t ** 2) / (2.0 * sigmas[j] ** 2)
            ) * np.cos(2.0 * np.pi * x_t / lambdas[j] + psis[j])
            np.testing.assert_allclose(actual[:, :, j], expected, atol=1e-6, rtol=0)

    def test_an_even_kernel_is_centered(self):
        """An even kernel_size centers on (k-1)/2, not on the off-centre k//2.

        With theta = psi = 0 the filter is even in both coordinates, so it must be
        invariant under a flip on both axes. The previous ``arange(k) - k // 2``
        grid gave [-2, -1, 0, 1] for k=4, breaking that symmetry.
        """
        bank = np.asarray(
            GaborFiltersInitializer(
                sweep="diagonal", theta_range=(0.0, 0.0), psi_range=(0.0, 0.0),
            )((4, 4, 1, 1))
        )[:, :, 0, 0]

        np.testing.assert_allclose(bank, bank[::-1, :], atol=1e-6, rtol=0)
        np.testing.assert_allclose(bank, bank[:, ::-1], atol=1e-6, rtol=0)

    def test_n_filters_tiles_the_bank(self):
        """n_filters < out_ch cycles the distinct filters across output channels."""
        bank = np.asarray(
            GaborFiltersInitializer(n_filters=4)((7, 7, 1, 12))
        )[:, :, 0, :]

        for j in range(4):
            np.testing.assert_array_equal(bank[:, :, j], bank[:, :, j + 4])
            np.testing.assert_array_equal(bank[:, :, j], bank[:, :, j + 8])

        assert not np.array_equal(bank[:, :, 0], bank[:, :, 1])

    # ------------------------------------------------------------------
    # dtype handling
    # ------------------------------------------------------------------

    def test_dtype_handling(self, initializer):
        """float32 / float64 honored; dtype=None -> keras.config.floatx()."""
        shape = (5, 5, 1, 4)

        for dtype in ("float32", "float64"):
            weights = initializer(shape, dtype=dtype)
            assert keras.backend.standardize_dtype(weights.dtype) == dtype

        weights_none = initializer(shape, dtype=None)
        assert keras.backend.standardize_dtype(weights_none.dtype) == keras.config.floatx()

    def test_float64_keeps_more_than_float32_precision(self):
        """The bank is computed in float64, so a float64 request is not a float32 cast."""
        shape = (11, 11, 1, 16)
        init = GaborFiltersInitializer()

        f64 = np.asarray(init(shape, dtype="float64"))
        f32 = np.asarray(init(shape, dtype="float32"))

        assert f64.dtype == np.float64
        # A float64 result that had been round-tripped through float32 would be
        # bit-identical to the float32 one after an upcast.
        assert not np.array_equal(f64, f32.astype(np.float64))
        np.testing.assert_allclose(f64, f32, atol=1e-6, rtol=0)

    # ------------------------------------------------------------------
    # Determinism
    # ------------------------------------------------------------------

    def test_determinism(self):
        """Separate instances and calls produce byte-identical arrays."""
        shape = (7, 7, 3, 12)

        a = GaborFiltersInitializer()
        b = GaborFiltersInitializer()

        w_a1 = np.asarray(a(shape))
        w_a2 = np.asarray(a(shape))
        w_b1 = np.asarray(b(shape))

        np.testing.assert_array_equal(w_a1, w_a2)
        np.testing.assert_array_equal(w_a1, w_b1)

    # ------------------------------------------------------------------
    # Filter-bank distribution
    # ------------------------------------------------------------------

    def test_channels_differ(self):
        """For out_ch > 1, at least two output channels differ."""
        shape = (7, 7, 1, 16)
        weights = np.asarray(GaborFiltersInitializer()(shape))

        out_ch = shape[3]
        distinct = False
        for j in range(1, out_ch):
            if not np.array_equal(weights[:, :, 0, j], weights[:, :, 0, 0]):
                distinct = True
                break
        assert distinct, "Expected at least two distinct output channels"

    def test_same_across_input_channels(self):
        """kernel[:, :, i, j] is identical across all input channels i."""
        shape = (5, 5, 3, 4)
        weights = np.asarray(GaborFiltersInitializer()(shape))

        out_ch = shape[3]
        for j in range(out_ch):
            np.testing.assert_array_equal(weights[:, :, 0, j], weights[:, :, 1, j])
            np.testing.assert_array_equal(weights[:, :, 0, j], weights[:, :, 2, j])

    def test_single_filter(self):
        """out_ch == 1 gives a finite, non-degenerate filter (not a range endpoint)."""
        shape = (11, 11, 1, 1)
        init = GaborFiltersInitializer()
        weights = np.asarray(init(shape))

        assert tuple(weights.shape) == shape
        assert np.all(np.isfinite(weights))

        # A single filter takes the MIDPOINT of every range, not the minimum:
        # the endpoints are exactly the degenerate extremes.
        sigmas, _, lambdas, gammas, _ = init._sweep_parameters(1, 11)
        resolved = init._resolved_ranges(11)
        assert sigmas[0] > resolved["sigma"][0]
        assert lambdas[0] > resolved["lambda"][0]
        assert gammas[0] > resolved["gamma"][0]

    # ------------------------------------------------------------------
    # get_config / from_config round-trip
    # ------------------------------------------------------------------

    def test_get_config_roundtrip(self):
        """get_config contains every constructor arg; from_config output is identical."""
        original = GaborFiltersInitializer(
            sigma_range=(1.0, 10.0),
            theta_range=(0.0, 180.0),
            lambda_range=(4.0, 50.0),
            gamma_range=(0.5, 2.0),
            psi_range=(0.0, 90.0),
            sweep="diagonal",
            n_filters=6,
            normalize=False,
        )
        config = original.get_config()

        for key in (
            "sigma_range",
            "theta_range",
            "lambda_range",
            "gamma_range",
            "psi_range",
            "sweep",
            "n_filters",
            "normalize",
        ):
            assert key in config

        reconstructed = GaborFiltersInitializer.from_config(config)

        assert tuple(reconstructed.sigma_range) == original.sigma_range
        assert tuple(reconstructed.theta_range) == original.theta_range
        assert tuple(reconstructed.lambda_range) == original.lambda_range
        assert tuple(reconstructed.gamma_range) == original.gamma_range
        assert tuple(reconstructed.psi_range) == original.psi_range
        assert reconstructed.sweep == original.sweep
        assert reconstructed.n_filters == original.n_filters
        assert reconstructed.normalize == original.normalize

        shape = (5, 5, 3, 8)
        np.testing.assert_array_equal(
            np.asarray(original(shape)),
            np.asarray(reconstructed(shape)),
        )

    def test_get_config_roundtrip_with_default_none_ranges(self):
        """The None scale ranges survive a config round-trip as None."""
        original = GaborFiltersInitializer()
        reconstructed = GaborFiltersInitializer.from_config(original.get_config())

        assert reconstructed.sigma_range is None
        assert reconstructed.lambda_range is None
        np.testing.assert_array_equal(
            np.asarray(original((7, 7, 2, 8))),
            np.asarray(reconstructed((7, 7, 2, 8))),
        )

    # ------------------------------------------------------------------
    # Keras serialization round-trip
    # ------------------------------------------------------------------

    def test_keras_serialization(self):
        """keras.initializers.serialize/deserialize reconstructs an equal init."""
        init = GaborFiltersInitializer(
            sigma_range=(1.5, 12.0),
            theta_range=(0.0, 270.0),
        )

        config = keras.initializers.serialize(init)
        deserialized = keras.initializers.deserialize(config)

        assert isinstance(deserialized, GaborFiltersInitializer)
        assert tuple(deserialized.sigma_range) == init.sigma_range
        assert tuple(deserialized.theta_range) == init.theta_range

        shape = (5, 5, 1, 8)
        np.testing.assert_array_equal(
            np.asarray(init(shape)),
            np.asarray(deserialized(shape)),
        )

    def test_keras_object_serialization(self):
        """serialize_keras_object/deserialize_keras_object round-trip."""
        init = GaborFiltersInitializer(gamma_range=(0.2, 1.0), sweep="diagonal")

        config = keras.saving.serialize_keras_object(init)
        deserialized = keras.saving.deserialize_keras_object(config)

        assert isinstance(deserialized, GaborFiltersInitializer)
        assert tuple(deserialized.gamma_range) == init.gamma_range
        assert deserialized.sweep == init.sweep

        shape = (5, 5, 2, 6)
        np.testing.assert_array_equal(
            np.asarray(init(shape)),
            np.asarray(deserialized(shape)),
        )

    # ------------------------------------------------------------------
    # Model save / load
    # ------------------------------------------------------------------

    def test_model_save_load(self):
        """A Conv2D model with the initializer saves and reloads via .keras."""
        inputs = keras.layers.Input(shape=(16, 16, 1), name="input")
        x = keras.layers.Conv2D(
            filters=8,
            kernel_size=5,
            kernel_initializer=GaborFiltersInitializer(),
            trainable=True,
            name="gabor_conv",
        )(inputs)
        x = keras.layers.GlobalAveragePooling2D()(x)
        outputs = keras.layers.Dense(5, activation="softmax", name="output")(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name="gabor_model")

        test_input = keras.random.normal([2, 16, 16, 1])
        original_prediction = model.predict(test_input, verbose=0)

        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, "gabor_model.keras")
            model.save(model_path)

            loaded_model = keras.models.load_model(
                model_path,
                custom_objects={"GaborFiltersInitializer": GaborFiltersInitializer},
            )

            loaded_prediction = loaded_model.predict(test_input, verbose=0)

            np.testing.assert_allclose(
                original_prediction,
                loaded_prediction,
                rtol=1e-6,
                atol=1e-6,
            )


class TestCreateGaborDepthwiseConv2D:
    """Test suite for the create_gabor_depthwise_conv2d builder utility.

    The depthwise builder applies the Gabor bank PER CHANNEL (no cross-channel
    mixing): output channels = in_channels * filters_per_channel.
    """

    def test_returns_depthwise_conv2d(self):
        """The builder returns a keras.layers.DepthwiseConv2D instance."""
        layer = create_gabor_depthwise_conv2d(filters_per_channel=8)
        assert isinstance(layer, keras.layers.DepthwiseConv2D)
        assert layer.depth_multiplier == 8

    def test_default_kernel_size(self):
        """kernel_size defaults to 11 (Ozbulak & Ekenel first-layer size)."""
        layer = create_gabor_depthwise_conv2d(filters_per_channel=8)
        assert layer.kernel_size == (11, 11)

    def test_default_activation_is_linear_and_bias_free(self):
        """activation defaults to None (linear passthrough); the stem stays bias-free."""
        layer = create_gabor_depthwise_conv2d(filters_per_channel=8)
        assert layer.activation is keras.activations.linear
        assert layer.use_bias is False

    def test_activation_is_honored(self):
        """A supplied activation reaches the underlying DepthwiseConv2D."""
        layer = create_gabor_depthwise_conv2d(filters_per_channel=4, activation="relu")
        assert layer.activation is keras.activations.relu

        # A rectified bank cannot emit negative responses.
        x = keras.random.normal([2, 16, 16, 3])
        y = np.asarray(layer(x))
        assert np.all(y >= 0.0)

    @pytest.mark.parametrize("filters_per_channel", [0, -1])
    def test_invalid_filters(self, filters_per_channel):
        """filters_per_channel < 1 raises ValueError."""
        with pytest.raises(ValueError):
            create_gabor_depthwise_conv2d(filters_per_channel=filters_per_channel)

    def test_filters_alias_is_accepted(self):
        """The deprecated `filters` alias still builds the same layer."""
        aliased = create_gabor_depthwise_conv2d(filters=8, kernel_size=7)
        current = create_gabor_depthwise_conv2d(filters_per_channel=8, kernel_size=7)

        assert aliased.depth_multiplier == current.depth_multiplier == 8
        assert aliased.kernel_size == current.kernel_size

    def test_filters_alias_conflict_raises(self):
        """Supplying both the alias and the current name is an error, not a silent win."""
        with pytest.raises(ValueError, match="deprecated"):
            create_gabor_depthwise_conv2d(filters_per_channel=8, filters=4)

    def test_missing_filters_raises(self):
        """Neither name supplied is a ValueError, not a TypeError from Keras."""
        with pytest.raises(ValueError, match="filters_per_channel"):
            create_gabor_depthwise_conv2d()

    def test_per_channel_output_no_mixing(self):
        """Output channels == in_channels * filters (no cross-channel mixing)."""
        filters = 8
        for in_ch in (1, 3):
            layer = create_gabor_depthwise_conv2d(
                filters_per_channel=filters, kernel_size=7
            )
            y = layer(np.zeros((1, 16, 16, in_ch), dtype="float32"))
            assert y.shape[-1] == in_ch * filters

    def test_kernel_matches_initializer(self):
        """Built layer's depthwise kernel equals the Gabor bank for (kh,kw,in,filters)."""
        filters = 8
        in_ch = 3
        kh, kw = 7, 7
        layer = create_gabor_depthwise_conv2d(
            filters_per_channel=filters, kernel_size=7
        )
        layer.build((None, 16, 16, in_ch))

        # Keras 3.8 exposes the depthwise weight as `.kernel` with shape
        # (kh, kw, in_ch, depth_multiplier).
        kernel = np.asarray(layer.kernel)
        assert kernel.shape == (kh, kw, in_ch, filters)
        expected = np.asarray(
            GaborFiltersInitializer()((kh, kw, in_ch, filters))
        )
        np.testing.assert_allclose(kernel, expected, atol=1e-6)
        # Same 2D bank replicated across input channels (per-channel application).
        for c in range(1, in_ch):
            np.testing.assert_allclose(kernel[:, :, 0, :], kernel[:, :, c, :], atol=1e-6)

    def test_trainable_flag_defaults_frozen(self):
        """trainable defaults to False (frozen front-end) and is honored."""
        frozen_layer = create_gabor_depthwise_conv2d(filters_per_channel=4)
        trainable_layer = create_gabor_depthwise_conv2d(
            filters_per_channel=4, trainable=True
        )
        assert frozen_layer.trainable is False
        assert trainable_layer.trainable is True

    def test_builder_model_save_load(self):
        """A model built with create_gabor_depthwise_conv2d round-trips through .keras."""
        inputs = keras.layers.Input(shape=(16, 16, 3), name="input")
        x = create_gabor_depthwise_conv2d(
            filters_per_channel=8,
            kernel_size=7,
            activation="relu",
            name="gabor_depthwise",
        )(inputs)
        x = keras.layers.GlobalAveragePooling2D()(x)
        outputs = keras.layers.Dense(5, activation="softmax", name="output")(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name="gabor_depthwise_model")

        test_input = keras.random.normal([2, 16, 16, 3])
        original_prediction = model.predict(test_input, verbose=0)

        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, "gabor_depthwise_model.keras")
            model.save(model_path)

            loaded_model = keras.models.load_model(
                model_path,
                custom_objects={"GaborFiltersInitializer": GaborFiltersInitializer},
            )

            loaded_prediction = loaded_model.predict(test_input, verbose=0)

            np.testing.assert_allclose(
                original_prediction,
                loaded_prediction,
                rtol=1e-6,
                atol=1e-6,
            )

            loaded_gabor = loaded_model.get_layer("gabor_depthwise")
            assert loaded_gabor.activation is keras.activations.relu


class TestCreateGaborConv2D:
    """Test suite for the create_gabor_conv2d builder (cross-channel warm start)."""

    def test_returns_conv2d_with_keras_filter_semantics(self):
        """`filters` means output channels here, unlike the depthwise builder."""
        layer = create_gabor_conv2d(filters=16, kernel_size=7)
        assert isinstance(layer, keras.layers.Conv2D)

        y = layer(np.zeros((1, 16, 16, 3), dtype="float32"))
        assert y.shape[-1] == 16

    def test_defaults_to_trainable(self):
        """The paper's use case is a trainable warm start, not a frozen transform."""
        assert create_gabor_conv2d(filters=4).trainable is True
        assert create_gabor_conv2d(filters=4, trainable=False).trainable is False

    def test_kernel_matches_initializer(self):
        """The built kernel is exactly the normalized Gabor bank."""
        layer = create_gabor_conv2d(filters=8, kernel_size=7)
        layer.build((None, 16, 16, 3))

        expected = np.asarray(GaborFiltersInitializer()((7, 7, 3, 8)))
        np.testing.assert_allclose(np.asarray(layer.kernel), expected, atol=1e-6)

    @pytest.mark.parametrize("filters", [0, -1])
    def test_invalid_filters(self, filters):
        """filters < 1 raises ValueError."""
        with pytest.raises(ValueError):
            create_gabor_conv2d(filters=filters)

    def test_gradients_flow_through_the_warm_start(self):
        """The warm-started kernel actually trains (this is why it is normalized)."""
        model = keras.Sequential([
            keras.layers.Input(shape=(16, 16, 1)),
            create_gabor_conv2d(filters=8, kernel_size=5, name="gabor"),
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(2),
        ])
        model.compile(optimizer=keras.optimizers.SGD(0.1), loss="mse")

        before = np.asarray(model.get_layer("gabor").kernel)
        model.fit(
            np.random.default_rng(0).normal(size=(8, 16, 16, 1)).astype("float32"),
            np.zeros((8, 2), dtype="float32"),
            epochs=1, verbose=0,
        )
        after = np.asarray(model.get_layer("gabor").kernel)

        assert not np.array_equal(before, after)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
