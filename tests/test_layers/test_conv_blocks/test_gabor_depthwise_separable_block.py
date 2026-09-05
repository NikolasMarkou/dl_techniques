"""Core battery for GaborDepthwiseSeparableBlock: validation, forward, shapes, layout, serialization."""

import os
import keras
import numpy as np
import pytest

from dl_techniques.layers.conv_blocks.gabor_depthwise_separable_block import (
    GaborDepthwiseSeparableBlock,
)

# ---------------------------------------------------------------------
# Shared constants. `K = 3` (not the layer default 11) keeps `padding='valid'`
# from consuming most of a 16x16 input, so the 'same' and 'valid' arms stay
# distinguishable at both strides.
# ---------------------------------------------------------------------

B, H, W = 2, 16, 16
K = 3
M = 2
F = 8

# The block's ENTIRE weight layout at the shipped defaults, keyed by `w.path`
# relative to the block. MEASURED, not predicted: both convolutions are
# `use_bias=False` at defaults, so there are exactly two variables and no bias
# entry. Step 1(e) of plan-2026-09-05T115518-e69163e4 recorded the same pair.
EXPECTED_WEIGHT_SUFFIXES = {
    "gabor_depthwise/kernel",
    "pointwise_conv/kernel",
}


@pytest.fixture
def rng():
    """Deterministic generator; every input in this module comes from seed 0."""
    return np.random.default_rng(0)


@pytest.fixture
def sample_input(rng):
    """A ``(B, H, W, 3)`` float32 batch."""
    return rng.standard_normal((B, H, W, 3)).astype("float32")


def _relative_weight_paths(block):
    """Return ``{w.path}`` with the block's own name prefix stripped off."""
    prefix = block.name + "/"
    return {
        w.path[len(prefix):] if w.path.startswith(prefix) else w.path
        for w in block.weights
    }


# ---------------------------------------------------------------------
# A. Construction + validation (SC-13)
# ---------------------------------------------------------------------

class TestConstructionAndValidation:
    """Every named invalid argument raises, and a valid layer starts unbuilt."""

    def test_default_construction_is_unbuilt(self):
        layer = GaborDepthwiseSeparableBlock(filters=F)
        assert layer.built is False
        assert layer.filters == F
        assert layer.filters_per_channel == 4
        assert layer.kernel_size == 11
        assert layer.strides == 1
        assert layer.padding == "same"
        # Defaults create NO optional stage at all (guide 16.3 layout claim).
        assert layer.gabor_norm is None
        assert layer.gabor_activation is None

    def test_invalid_filters_raises(self):
        with pytest.raises(ValueError, match=r"filters must be positive, got -3"):
            GaborDepthwiseSeparableBlock(filters=-3)

    def test_zero_filters_raises(self):
        with pytest.raises(ValueError, match=r"filters must be positive, got 0"):
            GaborDepthwiseSeparableBlock(filters=0)

    def test_invalid_filters_per_channel_raises(self):
        with pytest.raises(ValueError, match=r"filters_per_channel must be >= 1, got 0"):
            GaborDepthwiseSeparableBlock(filters=F, filters_per_channel=0)

    def test_invalid_kernel_size_int_raises(self):
        with pytest.raises(ValueError, match=r"kernel_size must be positive, got 0"):
            GaborDepthwiseSeparableBlock(filters=F, kernel_size=0)

    def test_invalid_kernel_size_tuple_raises(self):
        # The message quotes the ORIGINAL tuple, so the parens must be escaped.
        with pytest.raises(ValueError, match=r"kernel_size must be positive, got \(3, 0\)"):
            GaborDepthwiseSeparableBlock(filters=F, kernel_size=(3, 0))

    def test_invalid_strides_int_raises(self):
        with pytest.raises(ValueError, match=r"strides must be positive, got 0"):
            GaborDepthwiseSeparableBlock(filters=F, strides=0)

    def test_invalid_strides_tuple_raises(self):
        with pytest.raises(ValueError, match=r"strides must be positive, got \(0, 2\)"):
            GaborDepthwiseSeparableBlock(filters=F, strides=(0, 2))

    def test_invalid_padding_raises(self):
        with pytest.raises(ValueError, match=r"padding must be 'same' or 'valid', got 'reflect'"):
            GaborDepthwiseSeparableBlock(filters=F, padding="reflect")

    def test_rank3_input_raises(self, rng):
        layer = GaborDepthwiseSeparableBlock(filters=F, filters_per_channel=M, kernel_size=K)
        x = rng.standard_normal((B, H, 3)).astype("float32")
        with pytest.raises(ValueError, match=r"got shape with 3 dimensions"):
            layer(x)

    def test_rank5_input_raises(self, rng):
        layer = GaborDepthwiseSeparableBlock(filters=F, filters_per_channel=M, kernel_size=K)
        x = rng.standard_normal((B, H, W, 3, 2)).astype("float32")
        with pytest.raises(ValueError, match=r"got shape with 5 dimensions"):
            layer(x)


# ---------------------------------------------------------------------
# B. Forward pass (SC-10)
# ---------------------------------------------------------------------

# (strides, padding, expected spatial extent) for a 16x16 input and K = 3.
# Derived by hand, NOT by calling the layer's own helper:
#   same,  s=1 -> ceil(16/1) = 16      same,  s=2 -> ceil(16/2) = 8
#   valid, s=1 -> ceil((16-3+1)/1) = 14
#   valid, s=2 -> ceil((16-3+1)/2) = ceil(14/2) = 7
FORWARD_CASES = [
    (1, "same", 16),
    (2, "same", 8),
    (1, "valid", 14),
    (2, "valid", 7),
]


class TestForwardPass:
    """Shape and finiteness across strides x padding x input channels."""

    @pytest.mark.parametrize("strides,padding,out_hw", FORWARD_CASES)
    @pytest.mark.parametrize("in_channels", [1, 3])
    def test_forward_shape_and_finite(self, rng, strides, padding, out_hw, in_channels):
        layer = GaborDepthwiseSeparableBlock(
            filters=F,
            filters_per_channel=M,
            kernel_size=K,
            strides=strides,
            padding=padding,
        )
        x = rng.standard_normal((B, H, W, in_channels)).astype("float32")
        y = layer(x, training=False)

        assert tuple(y.shape) == (B, out_hw, out_hw, F)
        assert bool(keras.ops.all(keras.ops.isfinite(y)))

    def test_depthwise_kernel_shape_is_channelwise(self, sample_input):
        """The frozen bank is (kh, kw, in_channels, filters_per_channel)."""
        layer = GaborDepthwiseSeparableBlock(
            filters=F, filters_per_channel=M, kernel_size=K
        )
        y = layer(sample_input, training=False)
        assert bool(keras.ops.all(keras.ops.isfinite(y)))
        assert tuple(layer.gabor_depthwise.kernel.shape) == (K, K, 3, M)
        # The pointwise stage mixes channels * filters_per_channel -> filters.
        assert tuple(layer.pointwise_conv.kernel.shape) == (1, 1, 3 * M, F)


# ---------------------------------------------------------------------
# C. compute_output_shape (SC-14)
# ---------------------------------------------------------------------

class TestComputeOutputShape:
    """`compute_output_shape` is a pure function of stored config."""

    @pytest.mark.parametrize("strides,padding,out_hw", FORWARD_CASES)
    def test_unbuilt_compute_output_shape_matches_call(self, sample_input, strides, padding, out_hw):
        layer = GaborDepthwiseSeparableBlock(
            filters=F,
            filters_per_channel=M,
            kernel_size=K,
            strides=strides,
            padding=padding,
        )
        # Queried BEFORE any build: the whole point of the criterion.
        assert layer.built is False
        predicted = layer.compute_output_shape((B, H, W, 3))
        assert tuple(predicted) == (B, out_hw, out_hw, F)

        y = layer(sample_input, training=False)
        assert bool(keras.ops.all(keras.ops.isfinite(y)))
        assert tuple(y.shape) == tuple(predicted)

    def test_compute_output_shape_rejects_wrong_rank(self):
        layer = GaborDepthwiseSeparableBlock(filters=F, kernel_size=K)
        with pytest.raises(ValueError, match=r"Expected 4D input shape, got 3D"):
            layer.compute_output_shape((B, H, 3))

    def test_symbolic_none_spatial_dims(self, sample_input):
        """`keras.Input((None, None, C))` traces and keeps the static channel count."""
        inputs = keras.Input(shape=(None, None, 3))
        layer = GaborDepthwiseSeparableBlock(
            filters=F, filters_per_channel=M, kernel_size=K, name="sym_block"
        )
        outputs = layer(inputs)
        assert tuple(outputs.shape) == (None, None, None, F)
        assert tuple(layer.compute_output_shape((None, None, None, 3))) == (None, None, None, F)

        model = keras.Model(inputs, outputs)
        y = model(sample_input, training=False)
        assert tuple(y.shape) == (B, H, W, F)
        assert bool(keras.ops.all(keras.ops.isfinite(y)))


# ---------------------------------------------------------------------
# D. Build parity + no-extra-sub-layer (SC-5)
# ---------------------------------------------------------------------

class TestBuildParity:
    """The weight layout is pinned literally, and the OFF stages create nothing."""

    def test_default_weight_layout_is_pinned(self, sample_input):
        block = GaborDepthwiseSeparableBlock(
            filters=F, filters_per_channel=M, kernel_size=K, name="parity_block"
        )
        y = block(sample_input, training=False)
        assert bool(keras.ops.all(keras.ops.isfinite(y)))

        paths = _relative_weight_paths(block)
        assert paths == EXPECTED_WEIGHT_SUFFIXES
        # Both stages are bias-free at defaults; a bias entry here would mean the
        # block silently stopped being positively homogeneous.
        assert not any(p.endswith("/bias") for p in paths)
        assert len(block.weights) == 2

    def test_optional_stages_off_create_no_sublayer(self, sample_input):
        """`normalization_type=None, activation=None` materializes nothing extra."""
        block = GaborDepthwiseSeparableBlock(
            filters=F,
            filters_per_channel=M,
            kernel_size=K,
            normalization_type=None,
            activation=None,
            name="off_block",
        )
        assert block.gabor_norm is None
        assert block.gabor_activation is None

        y = block(sample_input, training=False)
        assert bool(keras.ops.all(keras.ops.isfinite(y)))

        # No extra weight appears relative to the pinned default layout.
        assert _relative_weight_paths(block) == EXPECTED_WEIGHT_SUFFIXES
        assert len(block.weights) == 2
        # And no extra sub-layer object is tracked either.
        assert block.gabor_norm is None
        assert block.gabor_activation is None


# ---------------------------------------------------------------------
# E. Config round trip (SC-4)
# ---------------------------------------------------------------------

# A NON-DEFAULT value for every one of the 19 constructor parameters. A round
# trip over defaults is the classic vacuous version of this test: a dropped key
# would still reconstruct the default and the configs would still compare equal.
NON_DEFAULT_KWARGS = dict(
    filters=7,                                  # required, still set explicitly
    filters_per_channel=2,                      # default 4
    kernel_size=5,                              # default 11
    strides=2,                                  # default 1
    padding="valid",                            # default 'same'
    sigma_range=(1.0, 2.0),                     # default None
    theta_range=(0.0, 90.0),                    # default (0.0, 180.0)
    lambda_range=(2.0, 4.0),                    # default None
    gamma_range=(0.6, 1.2),                     # default (0.5, 1.5)
    psi_range=(0.0, 180.0),                     # default (0.0, 360.0)
    sweep="diagonal",                           # default 'product'
    normalize=False,                            # default True
    normalization_type="layer_norm",            # default None
    normalization_kwargs={"epsilon": 1e-5},     # default None
    activation="relu",                          # default None
    activation_kwargs={"max_value": 6.0},       # default None
    pointwise_use_bias=True,                    # default False
    kernel_initializer="glorot_uniform",        # default 'he_normal'
)


class TestConfigRoundTrip:
    """`get_config()` -> `from_config()` reconstructs an equal-config layer."""

    def test_round_trip_over_every_non_default_parameter(self):
        layer = GaborDepthwiseSeparableBlock(
            kernel_regularizer=keras.regularizers.L2(1e-4),  # default None
            name="cfg_block",
            **NON_DEFAULT_KWARGS,
        )
        config = layer.get_config()

        # Anti-vacuity: every constructor parameter must actually be in the dict.
        expected_keys = set(NON_DEFAULT_KWARGS) | {"kernel_regularizer"}
        assert expected_keys <= set(config)

        rebuilt = GaborDepthwiseSeparableBlock.from_config(config)
        assert rebuilt.get_config() == config

        # Spot-check that the values survived as objects, not just as dict text.
        assert rebuilt.filters == 7
        assert rebuilt.sweep == "diagonal"
        assert rebuilt.normalize is False
        assert rebuilt.pointwise_use_bias is True
        assert rebuilt.gabor_norm is not None
        assert rebuilt.gabor_activation is not None
        assert rebuilt.kernel_regularizer is not None


# ---------------------------------------------------------------------
# F. `.keras` model save/load (SC-3)
# ---------------------------------------------------------------------

class TestKerasSaveLoad:
    """A full `.keras` round trip, compared on values AND on weights."""

    def test_saved_model_round_trip_is_exact(self, sample_input, tmp_path):
        inputs = keras.Input(shape=(H, W, 3))
        outputs = GaborDepthwiseSeparableBlock(
            filters=F, filters_per_channel=M, kernel_size=K, name="blk"
        )(inputs)
        model = keras.Model(inputs, outputs)

        y0 = model(sample_input, training=False)
        assert bool(keras.ops.all(keras.ops.isfinite(y0)))

        path = os.path.join(tmp_path, "gabor_dsb.keras")
        model.save(path)

        # MEASURED: `@register_dl_technique` makes `custom_objects` unnecessary --
        # the class resolves from the registry under
        # 'dl_techniques.layers.conv_blocks.gabor_depthwise_separable_block>GaborDepthwiseSeparableBlock'.
        # If a custom_objects mapping is ever needed here it must be keyed by
        # `keras.saving.get_registered_name(...)`, NEVER by the bare class name.
        loaded = keras.models.load_model(path)

        # Weight comparison FIRST, before the loaded model has ever been called:
        # after a call, lazily created state can mask a mismatch.
        original = {w.path: keras.ops.convert_to_numpy(w) for w in model.weights}
        restored = {w.path: keras.ops.convert_to_numpy(w) for w in loaded.weights}
        assert set(original) == set(restored)
        assert len(original) == 2
        for path_key in original:
            np.testing.assert_allclose(
                original[path_key], restored[path_key], rtol=0.0, atol=0.0,
                err_msg=f"weight {path_key} changed across the .keras round trip",
            )

        y1 = loaded(sample_input, training=False)
        assert bool(keras.ops.all(keras.ops.isfinite(y1)))
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(y0),
            keras.ops.convert_to_numpy(y1),
            rtol=0.0,
            atol=0.0,
            err_msg="reloaded model is not bit-identical to the original",
        )

        # The freeze must survive the round trip, not just construction.
        block = loaded.get_layer("blk")
        assert block.gabor_depthwise.trainable is False
        trainable_paths = {w.path for w in block.trainable_weights}
        assert block.gabor_depthwise.kernel.path not in trainable_paths
        assert len(trainable_paths) > 0

# ---------------------------------------------------------------------
