"""
Comprehensive test suite for HaarWaveletInitializer.

This module contains test cases for the HaarWaveletInitializer and
related functionality, ensuring proper wavelet properties, serialization,
and integration with Keras layers.

Several tests here are single-claim guards for defects found by review and
verified by measurement against the previous implementation:

* the bank was orthogonal but NOT orthonormal -- LL had norm 1 while LH/HL/HH
  had norm sqrt(2). Measured: Gram diagonal (1, 2, 2, 2), energy ratio 1.7487
  on iid input, per-sub-band variance (1.00, 2.01, 1.96, 2.02), and
  reconstruction by transpose failing with max error 4.04. Every claim of
  orthonormality, energy preservation and perfect reconstruction in the
  docstrings was therefore false. The previous suite pinned the defect: it
  asserted a total kernel energy of 7.0 (an orthonormal bank totals 4.0) under
  the name "Parseval's theorem".
* ``pattern_idx = (i * channel_multiplier + j) % 4`` made the sub-band of an
  output slot depend on the INPUT channel. Measured at channel_multiplier=2 with
  3 input channels: [[LL, LH], [HL, HH], [LL, LH]] -- input channel 1 had no
  low-pass path at all.
* ``super().__init__(**kwargs)`` could only raise, since
  ``keras.initializers.Initializer`` defines no ``__init__``.
* ``dtype=None`` was passed through to ``convert_to_tensor``, so under
  ``floatx='float64'`` the initializer silently returned float32.
* the builder validated ``input_shape`` for length and then never used it; a
  stride-2 'valid' decomposition on an odd size silently drops a row/column
  (measured: 255 -> 127 outputs, 129 -> 64).
"""

import pytest
import numpy as np
import keras
import tensorflow as tf
import tempfile
import os
from typing import Tuple, List
from unittest.mock import patch

from dl_techniques.initializers.haar_wavelet_initializer import (
    HAAR_PATTERNS,
    SUBBAND_NAMES,
    HaarWaveletInitializer,
    create_haar_depthwise_conv2d,
)


class TestHaarWaveletInitializer:
    """Test suite for HaarWaveletInitializer implementation."""

    @pytest.fixture
    def standard_shape(self) -> Tuple[int, int, int, int]:
        """Standard shape for 2x2 kernels with 3 input channels and 4 output channels."""
        return (2, 2, 3, 4)

    @pytest.fixture
    def single_channel_shape(self) -> Tuple[int, int, int, int]:
        """Shape for single input channel with 4 wavelet outputs."""
        return (2, 2, 1, 4)

    @pytest.fixture
    def initializer(self) -> HaarWaveletInitializer:
        """Default initializer instance."""
        return HaarWaveletInitializer(scale=1.0)

    @staticmethod
    def _get_expected_haar_patterns() -> List[np.ndarray]:
        """The four orthonormal 2D Haar patterns, written out independently.

        These are NOT read from HAAR_PATTERNS: an oracle that imports the value
        it is checking cannot fail. Every tap is +/- 0.5, which is what makes the
        Gram matrix the identity.
        """
        return [
            # LL: average both axes (approximation).
            np.array([[0.5, 0.5], [0.5, 0.5]]),
            # LH: average along height, difference along width.
            np.array([[0.5, -0.5], [0.5, -0.5]]),
            # HL: difference along height, average along width.
            np.array([[0.5, 0.5], [-0.5, -0.5]]),
            # HH: difference along both axes.
            np.array([[0.5, -0.5], [-0.5, 0.5]]),
        ]

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def test_initialization_defaults(self):
        """Test initialization with default parameters."""
        initializer = HaarWaveletInitializer()
        assert initializer.scale == 1.0

    def test_initialization_custom(self):
        """A custom scale is stored verbatim."""
        initializer = HaarWaveletInitializer(scale=2.0)
        assert initializer.scale == 2.0

    def test_invalid_scale_values(self):
        """Non-positive scales raise ValueError."""
        for scale in (0.0, -1.0, -0.5):
            with pytest.raises(ValueError, match="Scale must be positive"):
                HaarWaveletInitializer(scale=scale)

    def test_seed_is_accepted_and_inert(self):
        """`seed` is accepted for backward compatibility and changes nothing."""
        shape = (2, 2, 2, 4)
        np.testing.assert_array_equal(
            np.asarray(HaarWaveletInitializer(seed=42)(shape)),
            np.asarray(HaarWaveletInitializer(seed=None)(shape)),
        )

    def test_unknown_keyword_is_a_closed_signature(self):
        """The signature takes no **kwargs: an unknown key raises TypeError.

        keras.initializers.Initializer defines no __init__, so the previous
        `super().__init__(**kwargs)` could only ever raise an opaque
        "object.__init__() takes exactly one argument".
        """
        with pytest.raises(TypeError):
            HaarWaveletInitializer(not_a_parameter=1)

    # ------------------------------------------------------------------
    # __call__ shape and patterns
    # ------------------------------------------------------------------

    def test_call_standard_shape(self, initializer, standard_shape):
        """__call__ returns the requested shape with no NaN/Inf."""
        weights = np.asarray(initializer(standard_shape))

        assert weights.shape == standard_shape
        assert not np.any(np.isnan(weights))
        assert not np.any(np.isinf(weights))

    def test_call_single_channel(self, initializer, single_channel_shape):
        """The four output slots hold LL, LH, HL, HH in that order."""
        weights = np.asarray(initializer(single_channel_shape))
        expected_patterns = self._get_expected_haar_patterns()

        assert len(SUBBAND_NAMES) == 4
        for i in range(4):
            np.testing.assert_allclose(
                weights[:, :, 0, i], expected_patterns[i], atol=1e-6, rtol=0,
                err_msg=f"sub-band {SUBBAND_NAMES[i]} is wrong",
            )

    def test_the_module_constant_matches_the_documented_patterns(self):
        """HAAR_PATTERNS is the orthonormal basis the module docstring lists."""
        np.testing.assert_allclose(
            HAAR_PATTERNS, np.stack(self._get_expected_haar_patterns()),
            atol=1e-12, rtol=0,
        )

    @pytest.mark.parametrize("invalid_shape", [
        (3, 3, 3, 4),     # Invalid kernel size
        (2, 3, 3, 4),     # Mismatched dimensions
        (4, 4, 3, 4),     # Too large kernel
        (1, 1, 3, 4),     # Too small kernel
        (2, 2, 3),        # Wrong number of dimensions
        (2, 2, 3, 4, 1),  # Too many dimensions
    ])
    def test_invalid_shapes(self, invalid_shape, initializer):
        """Test that invalid shapes raise appropriate errors."""
        with pytest.raises(ValueError):
            initializer(invalid_shape)

    @pytest.mark.parametrize("shape", [(2, 2, 3, 0), (2, 2, 0, 4), (2, 2, 0, 0)])
    def test_non_positive_channel_counts_raise(self, initializer, shape):
        """A zero channel count used to hand back an empty kernel with no error."""
        with pytest.raises(ValueError, match=">= 1"):
            initializer(shape)

    # ------------------------------------------------------------------
    # Orthonormality -- the property the docstrings claim
    # ------------------------------------------------------------------

    def test_the_bank_is_orthonormal(self, single_channel_shape):
        """The Gram matrix is the IDENTITY, not merely diagonal.

        The previous bank was orthogonal with diagonal (1, 2, 2, 2): LH, HL and
        HH each carried twice the energy of LL.
        """
        weights = np.asarray(HaarWaveletInitializer(scale=1.0)(single_channel_shape))
        basis = weights[:, :, 0, :].reshape(4, 4).T  # rows = filters

        np.testing.assert_allclose(basis @ basis.T, np.eye(4), atol=1e-6, rtol=0)

    def test_energy_is_preserved(self, single_channel_shape):
        """Parseval: the analysis operator maps block energy to itself exactly.

        The previous implementation had a mean energy ratio of 1.7487 on iid
        input, and an input-dependent gain between 1x and 2x.
        """
        weights = np.asarray(HaarWaveletInitializer(scale=1.0)(single_channel_shape))
        basis = weights[:, :, 0, :].reshape(4, 4).T

        # Total kernel energy: an orthonormal 4-filter bank totals exactly 4.0
        # (the previous suite asserted 7.0 and called it Parseval's theorem).
        assert np.sum(weights ** 2) == pytest.approx(4.0, abs=1e-6)

        x = np.random.default_rng(0).normal(size=(4096, 4))
        coefficients = x @ basis.T
        np.testing.assert_allclose(
            (coefficients ** 2).sum(axis=1), (x ** 2).sum(axis=1), atol=1e-5, rtol=0
        )

    def test_every_subband_has_the_input_variance(self, single_channel_shape):
        """No sub-band drifts in scale; the previous detail bands were at 2x."""
        weights = np.asarray(HaarWaveletInitializer(scale=1.0)(single_channel_shape))
        basis = weights[:, :, 0, :].reshape(4, 4).T

        x = np.random.default_rng(1).normal(size=(40000, 4))
        variances = (x @ basis.T).var(axis=0)
        np.testing.assert_allclose(variances, 1.0, atol=0.05, rtol=0)

    def test_the_transpose_reconstructs(self, single_channel_shape):
        """Perfect reconstruction by the transpose, which orthogonality alone
        does not give: the previous bank needed the pseudo-inverse (max error
        4.04 through the transpose)."""
        weights = np.asarray(HaarWaveletInitializer(scale=1.0)(single_channel_shape))
        basis = weights[:, :, 0, :].reshape(4, 4).T

        x = np.random.default_rng(2).normal(size=(1024, 4))
        np.testing.assert_allclose(
            (x @ basis.T) @ basis, x, atol=1e-5, rtol=0
        )

    def test_orthogonality_properties(self, single_channel_shape):
        """Pairwise orthogonality holds for any positive scale."""
        for scale in (0.5, 1.0, 2.0):
            weights = np.asarray(
                HaarWaveletInitializer(scale=scale)(single_channel_shape)
            )
            patterns = [weights[:, :, 0, i].flatten() for i in range(4)]
            for i in range(4):
                for j in range(i + 1, 4):
                    np.testing.assert_allclose(
                        np.dot(patterns[i], patterns[j]), 0.0, atol=1e-6, rtol=0
                    )

    @pytest.mark.parametrize("scale,expected_max", [
        (1.0, 0.5),
        (2.0, 1.0),
        (0.5, 0.25),
        (4.0, 2.0),
    ])
    def test_scaling_behavior(self, scale, expected_max):
        """Every tap has magnitude 0.5 * scale.

        The previous implementation peaked at scale/sqrt(2), because three of the
        four filters were built at 1/sqrt(2) instead of 0.5.
        """
        weights = np.asarray(HaarWaveletInitializer(scale=scale)((2, 2, 1, 4)))
        assert np.max(np.abs(weights)) == pytest.approx(expected_max, abs=1e-6)
        assert np.min(np.abs(weights)) == pytest.approx(expected_max, abs=1e-6)

    def test_scaling_multiplies_energy_by_its_square(self):
        """scale != 1 preserves orthogonality, not normality -- state it as a fact."""
        base = np.asarray(HaarWaveletInitializer(scale=1.0)((2, 2, 1, 4)))
        scaled = np.asarray(HaarWaveletInitializer(scale=3.0)((2, 2, 1, 4)))
        assert np.sum(scaled ** 2) == pytest.approx(9.0 * np.sum(base ** 2), rel=1e-6)

    # ------------------------------------------------------------------
    # Sub-band layout
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("channel_multiplier", [1, 2, 3, 4, 5, 8])
    def test_every_input_channel_gets_the_same_bank(self, channel_multiplier):
        """Slot j holds sub-band j % 4 for EVERY input channel.

        The previous formula (i * channel_multiplier + j) % 4 made the sub-band
        depend on the input channel: at channel_multiplier=2 with 3 input
        channels the assignment was [[LL, LH], [HL, HH], [LL, LH]], leaving input
        channel 1 with no low-pass path.
        """
        in_channels = 3
        weights = np.asarray(
            HaarWaveletInitializer()((2, 2, in_channels, channel_multiplier))
        )
        expected_patterns = self._get_expected_haar_patterns()

        for i in range(in_channels):
            for j in range(channel_multiplier):
                np.testing.assert_allclose(
                    weights[:, :, i, j], expected_patterns[j % 4],
                    atol=1e-6, rtol=0,
                    err_msg=(
                        f"input channel {i}, slot {j} should hold "
                        f"{SUBBAND_NAMES[j % 4]}"
                    ),
                )

    def test_the_layout_is_addressable_by_slot_alone(self):
        """The slot -> sub-band map does not depend on the input channel index.

        This is the property downstream code needs in order to index a sub-band,
        and it is exactly what the old formula destroyed for cm != 4.
        """
        weights = np.asarray(HaarWaveletInitializer()((2, 2, 4, 3)))
        for j in range(3):
            for i in range(1, 4):
                np.testing.assert_array_equal(weights[:, :, 0, j], weights[:, :, i, j])

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def test_serialization_roundtrip(self):
        """get_config carries scale (and NOT the inert seed); from_config matches."""
        original = HaarWaveletInitializer(scale=1.5)
        config = original.get_config()

        assert config['scale'] == 1.5
        assert 'seed' not in config, "an inert argument must not be persisted"

        reconstructed = HaarWaveletInitializer.from_config(config)
        assert reconstructed.scale == original.scale

        shape = (2, 2, 3, 4)
        np.testing.assert_array_equal(
            np.asarray(original(shape)), np.asarray(reconstructed(shape))
        )

    def test_a_legacy_config_with_a_seed_still_loads(self):
        """Configs written before `seed` was retired must still deserialize."""
        restored = HaarWaveletInitializer.from_config({'scale': 2.0, 'seed': 42})
        assert restored.scale == 2.0

    def test_keras_serialization_compatibility(self):
        """keras.initializers.serialize/deserialize round-trip."""
        initializer = HaarWaveletInitializer(scale=2.0)
        deserialized = keras.initializers.deserialize(
            keras.initializers.serialize(initializer)
        )

        assert isinstance(deserialized, HaarWaveletInitializer)
        assert deserialized.scale == 2.0

    # ------------------------------------------------------------------
    # dtype
    # ------------------------------------------------------------------

    def test_dtype_handling(self, initializer, standard_shape):
        """An explicit dtype is honored."""
        for dtype in ('float32', 'float64'):
            weights = initializer(standard_shape, dtype=dtype)
            assert keras.backend.standardize_dtype(weights.dtype) == dtype

    def test_dtype_none_follows_floatx(self, initializer, standard_shape):
        """dtype=None resolves to keras.config.floatx().

        Previously `None` went straight to convert_to_tensor over a float32
        numpy buffer, so a float64 floatx silently got float32 back.
        """
        original = keras.config.floatx()
        try:
            for floatx in ('float32', 'float64'):
                keras.config.set_floatx(floatx)
                weights = initializer(standard_shape, dtype=None)
                assert keras.backend.standardize_dtype(weights.dtype) == floatx
        finally:
            keras.config.set_floatx(original)

    def test_float64_is_not_a_float32_upcast(self, initializer, standard_shape):
        """The buffer is allocated in the resolved dtype, not quantized first."""
        weights = np.asarray(initializer(standard_shape, dtype='float64'))
        assert weights.dtype == np.float64
        # 0.5 is exact in both dtypes, so compare the value that would reveal a
        # float32 round-trip on a scaled bank instead.
        scaled = np.asarray(
            HaarWaveletInitializer(scale=0.1)(standard_shape, dtype='float64')
        )
        assert not np.array_equal(scaled, scaled.astype(np.float32).astype(np.float64))

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def test_construction_does_not_log_at_info(self):
        """Construction is silent at INFO; it happens on every deserialization."""
        with patch('dl_techniques.initializers.haar_wavelet_initializer.logger') as mock_logger:
            HaarWaveletInitializer(scale=2.0)((2, 2, 1, 4))
            create_haar_depthwise_conv2d(input_shape=(32, 32, 3))
        mock_logger.info.assert_not_called()
        mock_logger.debug.assert_called()


class TestCreateHaarDepthwiseConv2D:
    """Test suite for the create_haar_depthwise_conv2d builder utility."""

    @pytest.fixture
    def input_shape(self) -> Tuple[int, int, int]:
        """A valid (even) input shape."""
        return (32, 32, 3)

    def test_basic_creation(self, input_shape):
        """The builder returns a correctly configured DepthwiseConv2D."""
        layer = create_haar_depthwise_conv2d(input_shape=input_shape)

        assert isinstance(layer, keras.layers.DepthwiseConv2D)
        assert layer.kernel_size == (2, 2)
        assert layer.strides == (2, 2)
        assert layer.padding == 'valid'
        assert layer.depth_multiplier == 4
        assert not layer.use_bias
        assert not layer.trainable

    def test_custom_parameters(self, input_shape):
        """Every knob reaches the layer."""
        layer = create_haar_depthwise_conv2d(
            input_shape=input_shape,
            channel_multiplier=8,
            scale=2.0,
            use_bias=True,
            kernel_regularizer='l2',
            trainable=True,
            name='custom_haar',
        )

        assert layer.depth_multiplier == 8
        assert layer.use_bias
        assert layer.trainable
        assert layer.name == 'custom_haar'
        assert layer.depthwise_regularizer is not None

    def test_scale_reaches_the_built_kernel(self, input_shape):
        """`scale` is not merely stored: it multiplies the built weights."""
        layer = create_haar_depthwise_conv2d(input_shape=input_shape, scale=2.0)
        layer.build((None,) + input_shape)

        assert np.max(np.abs(np.asarray(layer.kernel))) == pytest.approx(1.0, abs=1e-6)

    def test_forward_pass(self, input_shape):
        """Output is (B, H//2, W//2, C * channel_multiplier)."""
        layer = create_haar_depthwise_conv2d(input_shape=input_shape)
        output = layer(keras.random.normal([2] + list(input_shape)))

        assert output.shape == (2, 16, 16, 12)

    def test_wavelet_decomposition_properties(self, input_shape):
        """On a constant input only the LL slot of each channel responds."""
        layer = create_haar_depthwise_conv2d(input_shape=input_shape)
        output = np.asarray(layer(np.ones([1] + list(input_shape), dtype='float32')))

        for ch in range(input_shape[-1]):
            # LL of a constant 1.0 block: 4 * 0.5 = 2.0
            assert np.allclose(output[0, :, :, ch * 4], 2.0, atol=1e-6)
            for detail_idx in (1, 2, 3):
                assert np.allclose(
                    output[0, :, :, ch * 4 + detail_idx], 0.0, atol=1e-6
                )

    def test_the_output_channel_order_is_input_channel_major(self):
        """Output channel i*cm + j holds sub-band j of input channel i.

        Pinned on channel IDENTITY, not on shape: a transposed layout has the
        identical shape and would pass every dimension assertion.
        """
        layer = create_haar_depthwise_conv2d(input_shape=(4, 4, 3))
        x = np.zeros((1, 4, 4, 3), dtype='float32')
        x[..., 0], x[..., 1], x[..., 2] = 1.0, 10.0, 100.0

        y = np.asarray(layer(x))[0, 0, 0, :]
        np.testing.assert_allclose(
            y, [2.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 200.0, 0.0, 0.0, 0.0],
            atol=1e-5, rtol=0,
        )

    def test_the_layer_transform_preserves_energy(self):
        """End to end: the frozen layer is a Parseval frame on 2x2 blocks."""
        layer = create_haar_depthwise_conv2d(input_shape=(8, 8, 1))
        x = np.random.default_rng(3).normal(size=(4, 8, 8, 1)).astype('float32')
        y = np.asarray(layer(x))

        assert float((y ** 2).sum()) == pytest.approx(float((x ** 2).sum()), rel=1e-5)

    @pytest.mark.parametrize("invalid_input_shape", [(32, 32), (32, 32, 3, 1), (32,)])
    def test_invalid_input_shapes(self, invalid_input_shape):
        """input_shape must be 3D."""
        with pytest.raises(ValueError, match="Expected 3D input shape"):
            create_haar_depthwise_conv2d(input_shape=invalid_input_shape)

    @pytest.mark.parametrize("input_shape", [(255, 256, 3), (256, 255, 3), (129, 129, 1)])
    def test_odd_spatial_dimensions_are_rejected(self, input_shape):
        """A stride-2 'valid' decomposition would silently drop a row/column.

        Measured on the previous builder, which accepted these: 255 -> 127
        outputs (1 row discarded), 129 -> 64 (1 discarded).
        """
        with pytest.raises(ValueError, match="EVEN spatial"):
            create_haar_depthwise_conv2d(input_shape=input_shape)

    def test_unknown_spatial_dimensions_are_allowed(self):
        """A None spatial dimension cannot be checked and must not be rejected."""
        layer = create_haar_depthwise_conv2d(input_shape=(None, None, 3))
        assert layer.depth_multiplier == 4

    def test_invalid_channel_multiplier(self, input_shape):
        """channel_multiplier must be positive."""
        for channel_multiplier in (0, -1):
            with pytest.raises(ValueError, match="channel_multiplier must be positive"):
                create_haar_depthwise_conv2d(
                    input_shape=input_shape, channel_multiplier=channel_multiplier
                )

        layer = create_haar_depthwise_conv2d(
            input_shape=input_shape, channel_multiplier=2, trainable=False
        )
        assert layer.depth_multiplier == 2

    # ------------------------------------------------------------------
    # Warnings
    # ------------------------------------------------------------------

    def test_logging_warnings(self, input_shape):
        """A non-standard channel_multiplier on a frozen layer warns."""
        with patch('dl_techniques.initializers.haar_wavelet_initializer.logger') as mock_logger:
            create_haar_depthwise_conv2d(
                input_shape=input_shape, channel_multiplier=2, trainable=False
            )

        mock_logger.warning.assert_called()
        warning_call = mock_logger.warning.call_args[0][0]
        assert "channel_multiplier=2" in warning_call
        assert "standard wavelet decomposition" in warning_call

    def test_a_multiplier_above_four_warns_about_duplicates(self, input_shape):
        """cm > 4 cycles the 4-filter bank, so slots repeat bit-identically."""
        with patch('dl_techniques.initializers.haar_wavelet_initializer.logger') as mock_logger:
            create_haar_depthwise_conv2d(
                input_shape=input_shape, channel_multiplier=8, trainable=False
            )

        messages = " ".join(call[0][0] for call in mock_logger.warning.call_args_list)
        assert "duplicates" in messages

        # And the duplication is real, not just warned about.
        weights = np.asarray(HaarWaveletInitializer()((2, 2, 1, 8)))
        np.testing.assert_array_equal(weights[:, :, 0, :4], weights[:, :, 0, 4:])

    def test_a_regularizer_on_a_frozen_kernel_warns(self, input_shape):
        """Measured: Keras collects NO regularization loss from a frozen weight.

        model.losses is empty when trainable=False and holds one term when
        trainable=True, so a regularizer on the default frozen layer is a silent
        no-op rather than a constant penalty.
        """
        with patch('dl_techniques.initializers.haar_wavelet_initializer.logger') as mock_logger:
            frozen = create_haar_depthwise_conv2d(
                input_shape=input_shape,
                kernel_regularizer=keras.regularizers.L2(0.1),
                trainable=False,
            )
        messages = " ".join(call[0][0] for call in mock_logger.warning.call_args_list)
        assert "no-op" in messages

        trainable = create_haar_depthwise_conv2d(
            input_shape=input_shape,
            kernel_regularizer=keras.regularizers.L2(0.1),
            trainable=True,
        )
        for layer, expected_losses in ((frozen, 0), (trainable, 1)):
            model = keras.Sequential([keras.layers.Input(input_shape), layer])
            model(np.zeros((1,) + input_shape, dtype='float32'))
            assert len(model.losses) == expected_losses

    # ------------------------------------------------------------------
    # Integration
    # ------------------------------------------------------------------

    def test_model_integration(self, input_shape):
        """Test integration in a complete model."""
        inputs = keras.layers.Input(shape=input_shape)
        x = create_haar_depthwise_conv2d(input_shape=input_shape)(inputs)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.GlobalAveragePooling2D()(x)
        outputs = keras.layers.Dense(10, activation='softmax')(x)

        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'],
        )

        batch_size = 4
        prediction = model(
            keras.random.normal([batch_size] + list(input_shape)), training=False
        )

        assert prediction.shape == (batch_size, 10)
        assert np.allclose(np.sum(np.asarray(prediction), axis=1), 1.0, rtol=1e-6)

    def test_model_save_load(self, input_shape):
        """Test saving and loading a model with Haar wavelet layer."""
        inputs = keras.layers.Input(shape=input_shape, name='input')
        x = create_haar_depthwise_conv2d(
            input_shape=input_shape, name='haar_wavelet'
        )(inputs)
        x = keras.layers.GlobalAveragePooling2D()(x)
        outputs = keras.layers.Dense(5, activation='softmax', name='output')(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name='test_model')

        test_input = keras.random.normal([2] + list(input_shape))
        original_prediction = model.predict(test_input, verbose=0)

        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, "haar_model.keras")
            model.save(model_path)

            loaded_model = keras.models.load_model(
                model_path,
                custom_objects={'HaarWaveletInitializer': HaarWaveletInitializer},
            )

            loaded_prediction = loaded_model.predict(test_input, verbose=0)
            np.testing.assert_allclose(
                original_prediction, loaded_prediction, rtol=1e-6
            )

            haar_layer = loaded_model.get_layer('haar_wavelet')
            assert isinstance(haar_layer, keras.layers.DepthwiseConv2D)
            assert not haar_layer.trainable

    def test_gradient_flow(self, input_shape):
        """Test gradient flow through trainable Haar wavelet layer."""
        layer = create_haar_depthwise_conv2d(
            input_shape=input_shape, trainable=True
        )

        test_input = keras.Variable(keras.random.normal([1] + list(input_shape)))

        with tf.GradientTape() as tape:
            output = layer(test_input)
            loss = keras.ops.sum(keras.ops.square(output))

        gradients = tape.gradient(loss, layer.trainable_variables)

        assert gradients is not None
        assert len(gradients) == len(layer.trainable_variables)
        for grad in gradients:
            assert grad is not None
            assert not keras.ops.any(keras.ops.isnan(grad))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
