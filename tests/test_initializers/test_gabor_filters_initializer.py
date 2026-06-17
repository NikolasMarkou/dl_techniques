"""
Comprehensive test suite for GaborFiltersInitializer.

This module contains test cases for the GaborFiltersInitializer, ensuring
correct construction / validation, Gabor filter-bank distribution,
determinism, dtype handling, serialization round-trips, and integration
with Keras layers (Conv2D save/load).
"""

import os
import tempfile
from typing import Tuple

import keras
import numpy as np
import pytest

from dl_techniques.initializers import (
    GaborFiltersInitializer,
    create_gabor_conv2d,
)


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
        """Default ranges match Table I and are stored as tuples."""
        init = GaborFiltersInitializer()

        assert init.sigma_range == (2.0, 21.0)
        assert init.theta_range == (0.0, 360.0)
        assert init.lambda_range == (8.0, 100.0)
        assert init.gamma_range == (0.0, 300.0)
        assert init.psi_range == (0.0, 360.0)

        for attr in (
            init.sigma_range,
            init.theta_range,
            init.lambda_range,
            init.gamma_range,
            init.psi_range,
        ):
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
        """out_ch == 1 does not raise and returns a finite filter."""
        shape = (5, 5, 1, 1)
        weights = np.asarray(GaborFiltersInitializer()(shape))

        assert tuple(weights.shape) == shape
        assert np.all(np.isfinite(weights))

    # ------------------------------------------------------------------
    # get_config / from_config round-trip
    # ------------------------------------------------------------------

    def test_get_config_roundtrip(self):
        """get_config contains all 5 ranges; from_config output is identical."""
        original = GaborFiltersInitializer(
            sigma_range=(1.0, 10.0),
            theta_range=(0.0, 180.0),
            lambda_range=(4.0, 50.0),
            gamma_range=(0.5, 2.0),
            psi_range=(0.0, 90.0),
        )
        config = original.get_config()

        for key in (
            "sigma_range",
            "theta_range",
            "lambda_range",
            "gamma_range",
            "psi_range",
        ):
            assert key in config

        reconstructed = GaborFiltersInitializer.from_config(config)

        assert tuple(reconstructed.sigma_range) == original.sigma_range
        assert tuple(reconstructed.theta_range) == original.theta_range
        assert tuple(reconstructed.lambda_range) == original.lambda_range
        assert tuple(reconstructed.gamma_range) == original.gamma_range
        assert tuple(reconstructed.psi_range) == original.psi_range

        shape = (5, 5, 3, 8)
        np.testing.assert_array_equal(
            np.asarray(original(shape)),
            np.asarray(reconstructed(shape)),
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
        init = GaborFiltersInitializer(gamma_range=(0.0, 100.0))

        config = keras.saving.serialize_keras_object(init)
        deserialized = keras.saving.deserialize_keras_object(config)

        assert isinstance(deserialized, GaborFiltersInitializer)
        assert tuple(deserialized.gamma_range) == init.gamma_range

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


class TestCreateGaborConv2D:
    """Test suite for the create_gabor_conv2d builder utility."""

    def test_returns_conv2d(self):
        """The builder returns a keras.layers.Conv2D instance."""
        layer = create_gabor_conv2d(filters=8)
        assert isinstance(layer, keras.layers.Conv2D)
        assert layer.filters == 8

    @pytest.mark.parametrize("filters", [0, -1])
    def test_invalid_filters(self, filters):
        """filters < 1 raises ValueError."""
        with pytest.raises(ValueError):
            create_gabor_conv2d(filters=filters)

    def test_kernel_matches_initializer(self):
        """Built layer's kernel equals GaborFiltersInitializer()((kh,kw,in,filters))."""
        filters = 8
        in_ch = 3
        kh, kw = 5, 5
        layer = create_gabor_conv2d(filters=filters, kernel_size=5)

        # Build the layer on a known input shape (1, 16, 16, 3).
        layer.build((None, 16, 16, in_ch))

        kernel = np.asarray(layer.kernel)
        expected = np.asarray(
            GaborFiltersInitializer()((kh, kw, in_ch, filters))
        )

        np.testing.assert_allclose(kernel, expected, atol=1e-6)

    def test_trainable_flag(self):
        """trainable flag is honored on the returned layer."""
        trainable_layer = create_gabor_conv2d(filters=4, trainable=True)
        frozen_layer = create_gabor_conv2d(filters=4, trainable=False)

        assert trainable_layer.trainable is True
        assert frozen_layer.trainable is False

    def test_builder_model_save_load(self):
        """A model built with create_gabor_conv2d round-trips through .keras."""
        inputs = keras.layers.Input(shape=(16, 16, 1), name="input")
        x = create_gabor_conv2d(
            filters=8,
            kernel_size=5,
            trainable=True,
            name="gabor_conv",
        )(inputs)
        x = keras.layers.GlobalAveragePooling2D()(x)
        outputs = keras.layers.Dense(5, activation="softmax", name="output")(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name="gabor_builder_model")

        test_input = keras.random.normal([2, 16, 16, 1])
        original_prediction = model.predict(test_input, verbose=0)

        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, "gabor_builder_model.keras")
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
