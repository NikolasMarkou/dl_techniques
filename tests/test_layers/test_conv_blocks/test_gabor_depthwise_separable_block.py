"""Battery for GaborDepthwiseSeparableBlock.

Sections A-F are the core arms (validation, forward, shapes, layout,
serialization). Sections G-L are the identity / freeze / gradient guards --
the ones that make this a *Gabor*, *frozen* layer -- plus the homogeneity-set
drift guard and the ``call()`` rank re-check. Sections N-Q are the robustness
arms: dtype policies, XLA-vs-eager, the optional stages switched ON (the twins
of section D's OFF assertions), and degenerate spatial inputs.
"""

import os
import json
import re
import logging

import keras
import numpy as np
import pytest
# `tensorflow` is imported for ONE thing only: `tf.function(jit_compile=True)`
# in section O. Every other arm in this module is written against `keras.ops`
# so it stays backend-agnostic; XLA compilation is not, and cannot be.
import tensorflow as tf

from dl_techniques.initializers.gabor_filters_initializer import (
    GaborFiltersInitializer,
)
from dl_techniques.layers.conv_blocks.gabor_depthwise_separable_block import (
    GaborDepthwiseSeparableBlock,
    _POSITIVELY_HOMOGENEOUS_ACTIVATIONS,
)
from tests.test_models.gradient_flow_oracle import (
    assert_gradients_reach_every_trainable_weight,
)
# Section V's instrument. IMPORTED, not re-implemented: `_ContextPoisoner` and
# `StrictTrainingActivation` are this repo's established, PROVEN-RED probe for
# `training=` forwarding (provenance header at
# `tests/test_layers/test_attention/test_tripse_attention.py:627-645`). Read
# section V's block comment before touching either name.
from tests.test_layers.test_attention.test_tripse_attention import (
    _ContextPoisoner,
    StrictTrainingActivation,
)
import dl_techniques.layers.activations.factory as _act_factory

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
    """Return each ``w.path`` made genuinely RELATIVE to the block.

    The earlier form of this helper stripped ``block.name + "/"`` only when the
    path STARTED with it. That made every layout assertion in this module
    coupled to pytest COLLECTION ORDER, because Keras' global
    ``name_scope_stack`` is a process-global that nothing in the suite
    restores: `tests/test_layers/test_attention/test_tripse_attention.py`'s
    `TestTripSETrainingIsForwardedExplicitly::
    test_the_ambient_training_context_is_a_single_poisonable_slot` leaves
    ``'outer'`` on that stack permanently (root-caused in decisions.md D-018),
    so every variable created afterwards in the same process reads
    ``'outer/parity_block/gabor_depthwise/kernel'``. MEASURED at commit
    d5bcd6450 with the prefix-only strip: running that one node id before this
    module gave ``5 failed, 73 passed``, i.e. this module shipped five RED
    tests into the `tests/test_layers/` gate purely as an artefact of order.

    ``rpartition`` takes the suffix after the LAST occurrence of the block's
    own name, which is what plan step 3 asked for in the first place ("`w.path`
    suffixes relative to the block"). It hides nothing: an internal sub-layer
    rename still moves the suffix and still reddens the layout assertions --
    MEASURED, `gabor_depthwise` -> `gabor_dw` in the layer reddens
    `TestBuildParity::test_default_weight_layout_is_pinned` and
    `::test_optional_stages_off_create_no_sublayer` with this helper in place.

    The leak itself is NOT this plan's to fix (D-018): it belongs to
    `test_tripse_attention.py` and reddens six other packages too.
    """
    # DECISION plan-2026-09-05T115518-e69163e4/D-023: `rpartition`, NOT
    # `startswith`. Do NOT "simplify" this back to
    # `w.path[len(prefix):] if w.path.startswith(prefix) else w.path` -- that
    # is the form that shipped and it made this module's five layout
    # assertions fail whenever any earlier test in the process left a scope on
    # Keras' global `name_scope_stack`. Do NOT go the other way either and
    # strip everything before the last `/`: that would stop reddening on an
    # internal sub-layer rename, which is the whole point of the assertion.
    # See decisions.md D-023 (C-3) and D-018.
    prefix = block.name + "/"
    return {
        w.path.rpartition(prefix)[2] if prefix in w.path else w.path
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

    @pytest.mark.parametrize("strides", [(1, 2), (2, 1), (3, 2)])
    def test_non_square_strides_raises(self, strides):
        """MEASURED before the fix: `strides=(1, 2)` CONSTRUCTED, answered
        `compute_output_shape((1,16,16,3)) -> (1,14,6,8)`, and then died inside
        `call()` with a raw TF `InvalidArgumentError: Current implementation
        only supports equal length strides in the row and column dimensions.`
        The constructor now refuses it (review W-3, decisions.md D-023).
        """
        with pytest.raises(
            ValueError,
            match=r"strides must be equal in the height and width dimensions",
        ):
            GaborDepthwiseSeparableBlock(filters=F, strides=strides)

    @pytest.mark.parametrize("strides", [1, 2, (1, 1), (2, 2)])
    def test_square_strides_are_the_non_vacuous_control(self, strides):
        """The control for the arm above: every SQUARE form still constructs.

        Without this, `test_non_square_strides_raises` could be satisfied by a
        constructor that rejected `strides` outright.
        """
        layer = GaborDepthwiseSeparableBlock(filters=F, strides=strides)
        assert layer.strides == strides

    def test_non_square_kernel_size_is_still_accepted(self):
        """Only `strides` is square-constrained; `kernel_size` is NOT.

        The depthwise kernel may be rectangular -- TensorFlow's restriction is
        on the stride pair alone. Pinning this stops the W-3 fix from being
        widened into a rejection of rectangular Gabor windows.
        """
        layer = GaborDepthwiseSeparableBlock(filters=F, kernel_size=(3, 5))
        assert layer.kernel_size == (3, 5)

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

# An ODD input extent, for the same grid. 15 is divisible by NEITHER stride, so
# these four numbers are the ones a ceil/floor drift changes -- see
# `TestComputeOutputShape::test_odd_input_extent_matches_call`. Hand-derived:
#   same,  s=1 -> ceil(15/1) = 15       same,  s=2 -> ceil(15/2) = 8
#   valid, s=1 -> ceil((15-3+1)/1) = 13
#   valid, s=2 -> ceil(13/2) = 7
ODD = 15
ODD_INPUT_CASES = [
    (1, "same", 15),
    (2, "same", 8),
    (1, "valid", 13),
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

    def test_non_square_kernel_size_runs_and_has_the_predicted_shape(self, sample_input):
        """No arm in this module passed a TUPLE `kernel_size` before.

        That gap is why mutation ME (`_positive_pair` collapsing a non-square
        pair to `(pair[0], pair[0])`) survived the whole suite. Hand-derived for
        a 16x16 input, `kernel_size=(3, 5)`, stride 1, `'valid'`:
            h = 16 - 3 + 1 = 14,  w = 16 - 5 + 1 = 12
        i.e. the two axes MUST differ, so an implementation that reuses the
        height kernel for the width cannot pass.
        """
        layer = GaborDepthwiseSeparableBlock(
            filters=F,
            filters_per_channel=M,
            kernel_size=(3, 5),
            strides=1,
            padding="valid",
        )
        assert layer.built is False
        predicted = layer.compute_output_shape((B, H, W, 3))
        assert tuple(predicted) == (B, 14, 12, F)

        y = layer(sample_input, training=False)
        assert tuple(y.shape) == (B, 14, 12, F)
        assert bool(keras.ops.all(keras.ops.isfinite(y)))
        assert tuple(layer.gabor_depthwise.kernel.shape) == (3, 5, 3, M)

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

    @pytest.mark.parametrize("strides,padding,out_hw", ODD_INPUT_CASES)
    def test_odd_input_extent_matches_call(self, rng, strides, padding, out_hw):
        """The drift-visibility arm for review C-2 / criterion SC-19.

        Every other spatial arm in this module uses `H = W = 16`, which is
        divisible by BOTH strides tested, so a ceil-vs-floor error in the
        `'same'` branch is arithmetically invisible: `ceil(16/2) == 16 // 2`.
        MEASURED against the pre-fix code, a ceil->floor mutation left the suite
        at `63 passed`. `H = W = 15` is divisible by neither stride, so every
        expected value below differs under floor division (`same,s=2`: 8 vs 7;
        `valid,s=2`: 7 vs 6).

        Hand-derived, NOT read off the layer, for `ODD = 15` and `K = 3`:
            same,  s=1 -> ceil(15/1)      = 15    (floor: 15)
            same,  s=2 -> ceil(15/2)      = 8     (floor: 7)
            valid, s=1 -> ceil((15-3+1)/1) = 13   (floor: 13)
            valid, s=2 -> ceil(13/2)      = 7     (floor: 6)
        """
        layer = GaborDepthwiseSeparableBlock(
            filters=F,
            filters_per_channel=M,
            kernel_size=K,
            strides=strides,
            padding=padding,
        )
        assert layer.built is False
        predicted = layer.compute_output_shape((B, ODD, ODD, 3))
        assert tuple(predicted) == (B, out_hw, out_hw, F)

        x = rng.standard_normal((B, ODD, ODD, 3)).astype("float32")
        y = layer(x, training=False)
        assert tuple(y.shape) == tuple(predicted)
        assert bool(keras.ops.all(keras.ops.isfinite(y)))

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
# G. GABOR-IDENTITY (SC-9)
# ---------------------------------------------------------------------

# The seven Gabor-shaping kwargs, spelled LITERALLY at the layer's documented
# defaults. They are written out here rather than read off the block on
# purpose: the reference initializer below must be built from the block's
# CONSTRUCTOR ARGUMENTS, never from `block.gabor_depthwise.depthwise_initializer`
# or any other object reached through the block. Comparing a kernel against the
# initializer that produced it is self-referential and can never fail -- the
# exact vacuous shape this plan's pre-mortem names.
DEFAULT_GABOR_KWARGS = dict(
    sigma_range=None,
    theta_range=(0.0, 180.0),
    lambda_range=None,
    gamma_range=(0.5, 1.5),
    psi_range=(0.0, 360.0),
    sweep="product",
    normalize=True,
)

# A different value for every one of the seven. `sweep='diagonal'` is a real
# accepted value (`SWEEP_MODES` in gabor_filters_initializer.py; the initializer
# raises on anything else). This arm is what catches a kwarg the block forgets
# to thread through to `create_gabor_depthwise_conv2d`.
NON_DEFAULT_GABOR_KWARGS = dict(
    sigma_range=(1.0, 3.0),
    theta_range=(0.0, 90.0),
    lambda_range=(3.0, 6.0),
    gamma_range=(0.6, 1.2),
    psi_range=(0.0, 180.0),
    sweep="diagonal",
    normalize=False,
)

# The precedent tolerance, taken from `test_kernel_matches_initializer` in
# tests/test_initializers/test_gabor_filters_initializer.py:705-725
# (`np.testing.assert_allclose(kernel, expected, atol=1e-6)`). Reused, not
# invented.
GABOR_IDENTITY_ATOL = 1e-6


def _built_block(gabor_kwargs, name):
    """Build a defaults-shaped block on `(None, H, W, 3)` with the given Gabor kwargs."""
    block = GaborDepthwiseSeparableBlock(
        filters=F,
        filters_per_channel=M,
        kernel_size=K,
        name=name,
        **gabor_kwargs,
    )
    block.build((None, H, W, 3))
    return block


def _kernel(block):
    """The depthwise bank as numpy. Keras 3.8 exposes it as `.kernel`."""
    return np.asarray(keras.ops.convert_to_numpy(block.gabor_depthwise.kernel))


class TestGaborIdentity:
    """The built depthwise kernel IS the Gabor bank for the block's own kwargs."""

    def test_default_kwargs_kernel_is_the_gabor_bank(self):
        """Kernel == GaborFiltersInitializer(<defaults>)((K, K, 3, M)) at atol=1e-6.

        The reference is constructed from the literal kwargs the block was
        given, independently of the block, so a block that silently stopped
        using a Gabor initializer would redden this.
        """
        block = _built_block(DEFAULT_GABOR_KWARGS, "identity_default")
        kernel = _kernel(block)
        assert kernel.shape == (K, K, 3, M)

        expected = np.asarray(
            GaborFiltersInitializer(**DEFAULT_GABOR_KWARGS)((K, K, 3, M))
        )
        np.testing.assert_allclose(kernel, expected, atol=GABOR_IDENTITY_ATOL)

        # Same 2D bank replicated across input channels (per-channel application),
        # as the precedent test also asserts.
        for c in range(1, 3):
            np.testing.assert_allclose(
                kernel[:, :, 0, :], kernel[:, :, c, :], atol=GABOR_IDENTITY_ATOL
            )

    def test_negative_control_kernel_is_not_he_normal(self):
        """The comparison has discriminating power: a he_normal bank is far away.

        MEASURED margin at K=3, M=2, C=3, seed 1234: max|kernel - he_normal|
        = 0.9445, i.e. ~9.4e5 x the atol=1e-6 the positive arm uses. Without
        this arm, a comparison that had gone numerically degenerate (e.g. both
        sides all-zero) would still read green.
        """
        block = _built_block(DEFAULT_GABOR_KWARGS, "identity_negctl")
        kernel = _kernel(block)

        he = np.asarray(keras.initializers.HeNormal(seed=1234)((K, K, 3, M)))
        assert he.shape == kernel.shape
        max_abs_diff = float(np.max(np.abs(kernel - he)))
        assert max_abs_diff > 1e-2, (
            f"the Gabor bank is indistinguishable from a he_normal draw "
            f"(max|diff| = {max_abs_diff:g}); the identity assertion above has "
            f"no discriminating power"
        )
        # And the bank is not degenerate: it carries real structure.
        assert float(np.max(np.abs(kernel))) > 1e-3

    def test_non_default_kwargs_are_threaded_through(self):
        """All SEVEN Gabor kwargs reach the initializer, not just the defaults.

        A block that dropped any one of the seven on the way to
        `create_gabor_depthwise_conv2d` would build the DEFAULT bank; the second
        assertion below proves the default bank is a distinguishable object
        (MEASURED max|diff| = 1.4587 between the two banks), so this arm can
        actually fail.
        """
        block = _built_block(NON_DEFAULT_GABOR_KWARGS, "identity_nondefault")
        kernel = _kernel(block)

        expected = np.asarray(
            GaborFiltersInitializer(**NON_DEFAULT_GABOR_KWARGS)((K, K, 3, M))
        )
        np.testing.assert_allclose(kernel, expected, atol=GABOR_IDENTITY_ATOL)

        # Anti-vacuity: the non-default bank must NOT equal the default bank,
        # otherwise a dropped kwarg would be invisible to the assertion above.
        default_bank = np.asarray(
            GaborFiltersInitializer(**DEFAULT_GABOR_KWARGS)((K, K, 3, M))
        )
        assert float(np.max(np.abs(kernel - default_bank))) > 1e-2


# ---------------------------------------------------------------------
# H. TRAINABLE-SET (SC-7)
# ---------------------------------------------------------------------

class TestTrainableSet:
    """Only the pointwise kernel is trainable; the Gabor bank is not."""

    def test_gabor_kernel_absent_from_trainable_weights(self):
        block = _built_block(DEFAULT_GABOR_KWARGS, "trainable_set_block")

        trainable_paths = {w.path for w in block.trainable_weights}
        non_trainable_paths = {w.path for w in block.non_trainable_weights}

        # Anti-vacuity floor: an empty trainable set would satisfy "the Gabor
        # kernel is not in it" for free.
        assert len(block.trainable_weights) > 0

        assert block.gabor_depthwise.kernel.path not in trainable_paths
        assert block.pointwise_conv.kernel.path in trainable_paths
        assert block.gabor_depthwise.kernel.path in non_trainable_paths

        assert block.gabor_depthwise.trainable is False
        assert block.pointwise_conv.trainable is True
        # Exactly two weights at defaults, split one-and-one.
        assert len(block.weights) == 2
        assert len(block.trainable_weights) == 1
        assert len(block.non_trainable_weights) == 1


# ---------------------------------------------------------------------
# I. FROZENNESS after a real optimizer step (SC-8)
# ---------------------------------------------------------------------

def _one_step_model(name="frz_block", optimizer=None):
    """A functional model whose ONLY trainable weight is the block's 1x1 kernel."""
    block = GaborDepthwiseSeparableBlock(
        filters=F, filters_per_channel=M, kernel_size=K, name=name
    )
    inputs = keras.Input(shape=(H, W, 3))
    model = keras.Model(inputs, block(inputs))
    model.compile(
        optimizer=optimizer if optimizer is not None else keras.optimizers.Adam(1e-2),
        loss="mse",
    )
    return model, block


# DECISION plan-2026-09-05T115518-e69163e4/D-013: this class and
# `TestTrainableToggle` below are ONE pair; do not merge, and do not "fix" the
# scoping caveat away. Step 1(c) MEASURED that Keras 3.8's `trainable` setter
# recurses into sub-layers, so a real `False -> True` cycle DOES unfreeze the
# Gabor bank. Do NOT respond by adding a re-freeze hack in the layer's
# `build()`/`call()` (the plan pre-committed to stopping instead), and do NOT
# widen this class to cover the toggled regime -- it would then fail against
# correct, standard framework behaviour. See decisions.md D-013.
class TestFrozenness:
    """The depthwise bank does not move under a real optimizer step.

    SCOPE (decisions.md D-013, plan Conditional checkbox 3): this guard covers
    the DEFAULT, NEVER-TOGGLED path only -- a block constructed and trained
    without anyone assigning `trainable` on it or on an enclosing model. Step
    1(c) MEASURED that a real `False -> True` cycle on a parent DOES unfreeze
    the bank (standard Keras 3.8 semantics: the `trainable` setter recurses
    into `self._layers`). That regime is documented by `TestTrainableToggle`
    below, not by this class.

    The claim is about the WEIGHT, never about the gradient: a zero gradient is
    not a freeze under an adaptive optimizer.
    """

    def test_depthwise_kernel_is_bit_identical_after_one_step(self, rng):
        keras.utils.set_random_seed(0)
        model, block = _one_step_model("frz_block")

        x = rng.standard_normal((4, H, W, 3)).astype("float32")
        y = rng.standard_normal((4, H, W, F)).astype("float32")

        # `np.array(..., copy=True)`: a live Variable handle would track the
        # update and compare equal to itself no matter what happened.
        dw_before = np.array(
            keras.ops.convert_to_numpy(block.gabor_depthwise.kernel), copy=True
        )
        pw_before = np.array(
            keras.ops.convert_to_numpy(block.pointwise_conv.kernel), copy=True
        )

        model.train_on_batch(x, y)

        dw_after = np.array(
            keras.ops.convert_to_numpy(block.gabor_depthwise.kernel), copy=True
        )
        pw_after = np.array(
            keras.ops.convert_to_numpy(block.pointwise_conv.kernel), copy=True
        )

        # Anti-vacuity FIRST: if the pointwise kernel had not moved either, the
        # model learned nothing and "the Gabor kernel did not move" is free.
        # MEASURED movement under Adam(1e-2) after one step: 9.99975e-03.
        pointwise_movement = float(np.max(np.abs(pw_after - pw_before)))
        assert pointwise_movement > 0.0, (
            "the pointwise kernel did not move, so this step trained nothing "
            "and the frozenness assertion below is vacuous"
        )

        # Bit-identical, not within a tolerance.
        assert np.array_equal(dw_before, dw_after), (
            f"the frozen Gabor bank moved by "
            f"{float(np.max(np.abs(dw_after - dw_before))):g} under one "
            f"optimizer step"
        )


# ---------------------------------------------------------------------
# J. TRAINABLE-TOGGLE semantics (decisions.md D-013)
# ---------------------------------------------------------------------

class TestTrainableToggle:
    """Standard Keras `trainable` semantics, documented rather than wished away.

    Neither arm here reports a DEFECT of this block. Keras 3.8's
    `Layer.trainable` setter recurses into `self._layers`, so a real value
    change on a parent reaches the frozen sub-layer; and TF's
    `AutoTrackable.__setattr__` short-circuits `self.x = self.x` on identity, so
    re-assigning the value a parent already has never runs the setter at all.
    Both behaviours were MEASURED in step 1(c) of
    plan-2026-09-05T115518-e69163e4. They are asserted here so that the freeze
    guarantee's real boundary is in the test record, and so that a future Keras
    change to either behaviour is caught rather than silently absorbed.
    """

    def test_false_then_true_cycle_unfreezes_the_bank(self):
        """A REAL `False -> True` cycle on the block unfreezes the Gabor kernel."""
        block = _built_block(DEFAULT_GABOR_KWARGS, "toggle_block")
        assert block.gabor_depthwise.trainable is False
        assert len(block.trainable_weights) == 1

        block.trainable = False   # a real value change (the block was True)
        assert block.gabor_depthwise.trainable is False
        assert len(block.trainable_weights) == 0

        block.trainable = True    # a real value change back
        assert block.gabor_depthwise.trainable is True
        assert len(block.trainable_weights) == 2
        assert block.gabor_depthwise.kernel.path in {
            w.path for w in block.trainable_weights
        }

    def test_false_then_true_cycle_on_an_enclosing_model_unfreezes_the_bank(self):
        """The same cycle at `keras.Sequential` level, the regime a caller hits."""
        block = GaborDepthwiseSeparableBlock(
            filters=F, filters_per_channel=M, kernel_size=K, name="toggle_seq_block"
        )
        model = keras.Sequential([keras.layers.Input(shape=(H, W, 3)), block])
        assert len(model.trainable_weights) == 1

        model.trainable = False
        assert len(model.trainable_weights) == 0
        model.trainable = True

        assert block.gabor_depthwise.trainable is True
        assert len(model.trainable_weights) == 2

    def test_setting_trainable_true_on_an_already_true_parent_is_a_no_op(self):
        """The identity short-circuit: the bank stays frozen, as measured."""
        block = _built_block(DEFAULT_GABOR_KWARGS, "toggle_noop_block")
        assert block.trainable is True
        assert block.gabor_depthwise.trainable is False

        block.trainable = True   # NOT a value change -> setter never runs

        assert block.gabor_depthwise.trainable is False
        assert len(block.trainable_weights) == 1
        assert block.gabor_depthwise.kernel.path not in {
            w.path for w in block.trainable_weights
        }


# ---------------------------------------------------------------------
# K. GRADIENT FLOW, via the shared oracle (SC-6)
# ---------------------------------------------------------------------

class TestGradientFlow:
    """Every trainable weight receives a live gradient AFTER one real step.

    Run after a step, never at init: this repo's precedent is that asserting at
    initialization misreports dead weights. The frozen Gabor kernel needs no
    `expect_zero` waiver because it is not in `trainable_weights` at all -- and
    that claim is verified below rather than assumed, by showing the waiver
    ERRORS as an unmatched pattern.
    """

    def test_every_trainable_weight_is_live_after_one_step(self, rng):
        keras.utils.set_random_seed(0)
        model, block = _one_step_model("grad_block")

        x = rng.standard_normal((4, H, W, 3)).astype("float32")
        y = rng.standard_normal((4, H, W, F)).astype("float32")
        model.train_on_batch(x, y)

        report = assert_gradients_reach_every_trainable_weight(model, x)

        # Anti-vacuity floor: an empty report passes the oracle trivially.
        assert len(report) > 0
        assert len(model.trainable_weights) > 0
        assert len(report) == len(model.trainable_weights)

        pw_path = block.pointwise_conv.kernel.path
        assert pw_path in report, (
            f"the pointwise kernel {pw_path!r} is not in the gradient report "
            f"{sorted(report)}"
        )
        assert report[pw_path] is not None
        assert report[pw_path] > 0.0

        # The frozen bank never enters the report at all.
        assert block.gabor_depthwise.kernel.path not in report

    def test_a_waiver_for_the_frozen_bank_is_rejected_as_unmatched(self, rng):
        """Verifies the "no waiver needed" claim instead of assuming it.

        `expect_zero` is two-sided: every pattern must match a weight in the
        report. The frozen Gabor kernel is absent from `trainable_weights`, so a
        waiver naming it is stale by construction and the oracle says so.
        """
        keras.utils.set_random_seed(0)
        model, _ = _one_step_model("grad_waiver_block")

        x = rng.standard_normal((4, H, W, 3)).astype("float32")
        y = rng.standard_normal((4, H, W, F)).astype("float32")
        model.train_on_batch(x, y)

        with pytest.raises(AssertionError, match=r"matched no weight"):
            assert_gradients_reach_every_trainable_weight(
                model, x, expect_zero=("gabor_depthwise/kernel",)
            )


# ---------------------------------------------------------------------
# L. Homogeneity allowlist: drift + warning behaviour
# ---------------------------------------------------------------------

# The warning text is emitted through `dl_techniques.utils.logger`, which is
# `logging.getLogger("dl")` -- hence `logger="dl"` on every `caplog.at_level`
# below, matching the precedent in tests/test_models/test_sam2/test_neck.py:1076.
_WARN_MARKER = "positively-homogeneous allowlist"


class TestHomogeneityAllowlist:
    """The private copy has not drifted, and the warning fires exactly when it should."""

    def test_private_copy_is_a_subset_of_the_convunext_allowlist(self):
        """Drift guard for the COPY that D-007 deliberately keeps private.

        The import is deferred into the test body on purpose: the layer module
        must never import upward from `models/`. MEASURED import cost: 0.046 s
        with keras already loaded, so the plan's "drop it if heavy" escape
        hatch does not fire.
        """
        from dl_techniques.models.vision.convunext.model import (
            POSITIVELY_HOMOGENEOUS_ACTIVATIONS,
        )

        assert _POSITIVELY_HOMOGENEOUS_ACTIVATIONS <= POSITIVELY_HOMOGENEOUS_ACTIVATIONS
        # Anti-vacuity: the empty set is a subset of everything.
        assert len(_POSITIVELY_HOMOGENEOUS_ACTIVATIONS) > 0
        assert None in _POSITIVELY_HOMOGENEOUS_ACTIVATIONS
        assert "relu" in _POSITIVELY_HOMOGENEOUS_ACTIVATIONS

    def test_gelu_warns(self, caplog):
        with caplog.at_level(logging.WARNING, logger="dl"):
            GaborDepthwiseSeparableBlock(
                filters=F, filters_per_channel=M, kernel_size=K, activation="gelu"
            )
        messages = [r.getMessage() for r in caplog.records]
        assert any(_WARN_MARKER in m for m in messages), (
            f"no allowlist warning for activation='gelu'; captured: {messages}"
        )
        assert any("gelu" in m for m in messages)

    @pytest.mark.parametrize("activation", ["relu", None])
    def test_allowlisted_activation_does_not_warn(self, caplog, activation):
        """No warning for an allowlisted activation -- with the capture proven live.

        The `'gelu'` construction at the top is the instrument check: it forces
        a record into `caplog` inside this very context, so a no-warn assertion
        that passed only because the capture was misconfigured (wrong logger
        name, wrong level) cannot read green here.
        """
        with caplog.at_level(logging.WARNING, logger="dl"):
            GaborDepthwiseSeparableBlock(
                filters=F, filters_per_channel=M, kernel_size=K, activation="gelu"
            )
            assert any(_WARN_MARKER in r.getMessage() for r in caplog.records), (
                "the caplog capture is not seeing the 'dl' logger's warnings, so "
                "the no-warn assertion below would be vacuous"
            )

            caplog.clear()
            GaborDepthwiseSeparableBlock(
                filters=F,
                filters_per_channel=M,
                kernel_size=K,
                activation=activation,
            )

        messages = [r.getMessage() for r in caplog.records]
        assert not any(_WARN_MARKER in m for m in messages), (
            f"activation={activation!r} is allowlisted but warned: {messages}"
        )


# ---------------------------------------------------------------------
# M. The `call()` rank re-check (decisions.md D-017)
# ---------------------------------------------------------------------

# DECISION plan-2026-09-05T115518-e69163e4/D-017: the `match=` below is
# deliberately the CALL-ONLY wording `got shape (2, 8, 3)`. Do NOT relax it to
# the shared prefix "Expected 4D input" or to "4D": `build()` raises that same
# prefix (with `got shape with N dimensions: ...`) and the arm would then pass
# against the build-time check, i.e. become the guard-that-cannot-fail this
# whole class exists to retire. Do not delete the layer's `call()` re-assert
# either. See decisions.md D-017.
class TestCallRankRecheck:
    """D-017 resolved: the `call()` re-check IS reachable, on a PRE-BUILT layer.

    Step 3 recorded that the rank arms in `TestConstructionAndValidation` never
    reach it -- Keras builds before it calls, so `build()`'s check fires first
    with the distinct message `got shape with N dimensions`. That made the
    `call()` re-assert a guard that could not fail on the tested paths.

    MEASURED here: a layer already built on a rank-4 shape and then called with
    a rank-3 or rank-5 tensor raises from `call()` itself, with the different
    message `got shape (2, 8, 3)` (no "with N dimensions"). The arms below match
    on that call-only wording, so they cannot be satisfied by the `build()`
    check, and D-017's branch is now genuinely exercised rather than labelled
    unreachable.
    """

    @pytest.mark.parametrize(
        "bad_shape",
        [(2, 8, 3), (2, 8, 8, 3, 2)],
        ids=["rank3", "rank5"],
    )
    def test_prebuilt_layer_rejects_a_different_rank_from_call(self, bad_shape):
        layer = GaborDepthwiseSeparableBlock(
            filters=F, filters_per_channel=M, kernel_size=K, name="recheck_block"
        )
        rng_local = np.random.default_rng(0)
        layer(rng_local.standard_normal((2, 8, 8, 3)).astype("float32"))
        assert layer.built is True

        shape_text = re.escape(f"got shape {bad_shape}")
        with pytest.raises(ValueError, match=shape_text):
            layer(np.zeros(bad_shape, dtype="float32"))


# ---------------------------------------------------------------------
# N. dtype policies (SC-11)
# ---------------------------------------------------------------------

# What each global policy is EXPECTED to produce, keyed by policy name:
#   (layer.compute_dtype, layer.variable_dtype, numpy dtype of the output)
#
# MEASURED on keras 3.8 / TF 2.18, CPU-only, before this table was written --
# not predicted from the policy name. Pasted from the probe:
#
#   policy=float32        compute=float32  variable=float32  out=float32
#   policy=mixed_float16  compute=float16  variable=float32  out=float16
#   policy=float64        compute=float64  variable=float64  out=float64
#
# The middle row is the whole point of the table: under `mixed_float16` a Keras
# layer computes in half precision while its VARIABLES stay float32, so
# `compute_dtype != variable_dtype` there and nowhere else. Without this table
# the three parametrizations would run identical float32 arithmetic three times
# and assert identical numbers -- an inert policy arm, which is a defect this
# repo has already shipped once (`plans/LESSONS.md`).
_EXPECTED_POLICY_DTYPES = {
    "float32": ("float32", "float32", "float32"),
    "mixed_float16": ("float16", "float32", "float16"),
    "float64": ("float64", "float64", "float64"),
}


class TestDtypePolicies:
    """Construct, build and forward under float32 / mixed_float16 / float64.

    The global policy is set and restored by the `dtype_policy` fixture in
    `tests/test_layers/conftest.py`, which is parametrized over exactly these
    three policies and restores the previous one in a `finally`. That restore
    is deliberately centralised there (see that module's docstring: the third
    copy-pasted `try/finally` is the one that forgets), so nothing here
    touches `keras.mixed_precision.set_global_policy` directly.
    """

    def test_compute_and_output_dtypes_follow_the_policy(self, dtype_policy, rng):
        expected_compute, expected_variable, expected_out = _EXPECTED_POLICY_DTYPES[
            dtype_policy
        ]

        block = GaborDepthwiseSeparableBlock(
            filters=F,
            filters_per_channel=M,
            kernel_size=K,
            name=f"dtype_{dtype_policy}",
        )
        x = rng.standard_normal((B, H, W, 3)).astype("float32")
        y = block(x, training=False)

        # Finiteness and shape -- true under every policy, and therefore NOT on
        # its own evidence that the policy did anything.
        assert tuple(y.shape) == (B, H, W, F)
        assert bool(keras.ops.all(keras.ops.isfinite(y)))

        # The part that actually DIFFERS between the three parametrizations.
        assert block.compute_dtype == expected_compute
        assert block.variable_dtype == expected_variable
        assert keras.ops.convert_to_numpy(y).dtype.name == expected_out
        # Half precision is the only policy where the two diverge; asserting
        # the divergence itself keeps the mixed arm from silently degrading
        # into a second float32 run.
        assert (block.compute_dtype != block.variable_dtype) == (
            dtype_policy == "mixed_float16"
        )

    def test_gabor_bank_stays_frozen_under_every_policy(self, dtype_policy, rng):
        """A dtype change must not silently unfreeze the bank."""
        block = GaborDepthwiseSeparableBlock(
            filters=F,
            filters_per_channel=M,
            kernel_size=K,
            name=f"frozen_{dtype_policy}",
        )
        block(rng.standard_normal((B, H, W, 3)).astype("float32"), training=False)

        gabor_kernel = block.gabor_depthwise.kernel
        trainable_paths = {w.path for w in block.trainable_weights}

        assert block.gabor_depthwise.trainable is False
        assert gabor_kernel.trainable is False
        assert gabor_kernel.path not in trainable_paths
        # Anti-vacuity floor: the set is non-empty, so "not in" is a real claim.
        assert len(block.trainable_weights) == 1
        assert block.pointwise_conv.kernel.path in trainable_paths
        # Variables carry the policy's VARIABLE dtype, not its compute dtype.
        _, expected_variable, _ = _EXPECTED_POLICY_DTYPES[dtype_policy]
        assert gabor_kernel.dtype == expected_variable


# ---------------------------------------------------------------------
# O. XLA (`jit_compile=True`) versus eager (SC-12)
# ---------------------------------------------------------------------

class TestXlaVersusEager:
    """The compiled and interpreted paths agree to the dtype's resolution.

    TF32 is deliberately NOT touched here. `tests/test_layers/conftest.py`
    hosts a module-scoped `tf32_disabled` fixture and an autouse
    `_tf32_leak_canary`, and calling `enable_tensor_float_32_execution`
    inline would trip the canary for the next test in the session. The opt-in
    was checked rather than assumed: the max abs difference below was MEASURED
    twice, once with the process-global TF32 flag at its session default
    (`True`) and once with it off, and both runs gave exactly
    4.76837158203125e-07. TF32 is a GPU tensor-core path and these runs are
    CPU-only (`CUDA_VISIBLE_DEVICES=""`), so it has no numeric effect here and
    the module does not need `pytestmark`.
    """

    def test_jit_compiled_output_matches_eager(self, sample_input):
        block = GaborDepthwiseSeparableBlock(
            filters=F, filters_per_channel=M, kernel_size=K, name="jit_block"
        )
        # Build eagerly first, so both paths run against the SAME weights.
        y_eager = keras.ops.convert_to_numpy(block(sample_input, training=False))

        @tf.function(jit_compile=True)
        def compiled(t):
            return block(t, training=False)

        y_jit = np.asarray(compiled(tf.constant(sample_input)))

        assert y_jit.shape == y_eager.shape
        assert np.all(np.isfinite(y_jit))

        # Anti-vacuity: two all-zero tensors would agree to any tolerance.
        peak = float(np.max(np.abs(y_eager)))
        assert peak > 0.1, f"output is degenerate (max |y| = {peak}), comparison is vacuous"

        # TOLERANCE DERIVATION (not pasted -- derived here, from the dtype's
        # resolution and this layer's accumulation length):
        #   * float32 has eps = 2**-23 = 1.1920928955078125e-07 (relative).
        #   * The block accumulates over K*K = 3*3 = 9 taps in the depthwise
        #     stage and over in_channels * filters_per_channel = 3*2 = 6 taps
        #     in the pointwise stage, i.e. n = 15 floating-point additions on
        #     the longest path from an input element to an output element.
        #   * Worst-case accumulated rounding for n sequential additions is
        #     bounded by ~n * eps in relative terms, so in absolute terms by
        #     n * eps * max|y|. XLA is free to reassociate and to fuse the
        #     multiply-adds, which changes WHICH roundings happen but not this
        #     bound.
        #   => atol = 15 * 1.1920929e-07 * max|y|.
        # With the measured max|y| = 4.1019325 that is 7.33e-06.
        #
        # MEASURED max abs difference on this exact configuration:
        #   4.76837158203125e-07
        # which is exactly ONE ulp at magnitude 4.1 (for x in [4, 8) the float32
        # ulp is 2**(2-23) = 4.76837158203125e-07) -- i.e. the two paths differ
        # by a single last-place bit. The derived tolerance therefore carries
        # ~15x headroom over what was observed; both numbers are stated so a
        # future reader can see the margin rather than trust it.
        n_accumulations = K * K + 3 * M
        atol = n_accumulations * float(np.finfo(np.float32).eps) * peak
        max_abs_diff = float(np.max(np.abs(y_jit - y_eager)))
        assert max_abs_diff <= atol, (
            f"jit vs eager differ by {max_abs_diff}, above the derived "
            f"tolerance {atol} (n={n_accumulations}, max|y|={peak})"
        )


# ---------------------------------------------------------------------
# P. Optional stages ON -- the twins of section D's OFF assertions
#    (guide 16.3: every "nothing changed" assertion needs its twin)
# ---------------------------------------------------------------------

# MEASURED weight layouts with a normalization stage switched on, keyed by
# `normalization_type`. Both keys are real entries of the norms factory's
# `_TYPE_TO_CLASS` (18 keys, enumerated in step 1(b) of this plan). The layouts
# were read off built layers, not predicted from the class name.
_NORM_ON_LAYOUTS = {
    "layer_norm": {
        "gabor_depthwise/kernel",
        "gabor_norm/gamma",
        "gabor_norm/beta",
        "pointwise_conv/kernel",
    },
    "rms_norm": {
        "gabor_depthwise/kernel",
        "gabor_norm/scale",
        "pointwise_conv/kernel",
    },
}


class TestOptionalStagesOn:
    """Each optional stage, when switched on, really exists and really runs."""

    @pytest.mark.parametrize("normalization_type", sorted(_NORM_ON_LAYOUTS))
    def test_normalization_on_adds_a_sublayer_and_its_weights(
        self, sample_input, normalization_type
    ):
        block = GaborDepthwiseSeparableBlock(
            filters=F,
            filters_per_channel=M,
            kernel_size=K,
            normalization_type=normalization_type,
            name=f"norm_on_{normalization_type}",
        )
        assert block.gabor_norm is not None

        y = block(sample_input, training=False)
        assert tuple(y.shape) == (B, H, W, F)
        assert bool(keras.ops.all(keras.ops.isfinite(y)))

        paths = _relative_weight_paths(block)
        assert paths == _NORM_ON_LAYOUTS[normalization_type]
        # Strictly MORE weights than the defaults arm pinned in section D.
        assert paths > EXPECTED_WEIGHT_SUFFIXES
        assert len(block.weights) > len(EXPECTED_WEIGHT_SUFFIXES)
        # The norm carries its own `w.path` under the block, i.e. it is a
        # tracked sub-layer rather than a functional call.
        assert any(p.startswith("gabor_norm/") for p in paths)

        # S-3: the norm came from `create_normalization_layer`, whose epsilon
        # default is 1e-6. A bare `keras.layers.LayerNormalization()` would
        # carry 1e-3 -- a 1000x gap, MEASURED in step 1(b) of this plan. This
        # assertion is what would redden if someone "simplified" the factory
        # call away.
        assert block.gabor_norm.epsilon == pytest.approx(1e-6)

    def test_activation_relu_takes_the_registry_path(self, sample_input):
        """`'relu'` IS an `ACTIVATION_REGISTRY` key, so it reaches the factory."""
        block = GaborDepthwiseSeparableBlock(
            filters=F,
            filters_per_channel=M,
            kernel_size=K,
            activation="relu",
            name="act_relu",
        )
        assert block.gabor_activation is not None
        assert isinstance(block.gabor_activation, keras.layers.ReLU)

        y = block(sample_input, training=False)
        assert tuple(y.shape) == (B, H, W, F)
        assert bool(keras.ops.all(keras.ops.isfinite(y)))
        # An activation stage is weightless, so the layout is unchanged -- the
        # sub-layer OBJECT is the evidence here, not a new `w.path`.
        assert _relative_weight_paths(block) == EXPECTED_WEIGHT_SUFFIXES

    @pytest.mark.parametrize("activation", ["linear", "leaky_relu"])
    def test_non_registry_activations_route_through_the_keras_fallback(
        self, sample_input, activation
    ):
        """This arm is what proves decisions.md D-012's fix actually works.

        Step 1(a) MEASURED that `'linear'` and `'leaky_relu'` are NOT keys of
        `ACTIVATION_REGISTRY` (24 keys, `'relu'` among them, these two not) and
        that `create_activation_layer` raises `ValueError` on both. They are
        also two of the four names in the block's own positive-homogeneity
        allowlist, so a regression from `resolve_activation_layer` back to
        `create_activation_layer` would make the block raise on half the
        activations it most wants to support. Without this arm that regression
        is invisible: the `'relu'` arm above would stay green.

        `keras.layers.Activation` (as opposed to `keras.layers.ReLU`) is the
        observable signature of the fallback path.
        """
        block = GaborDepthwiseSeparableBlock(
            filters=F,
            filters_per_channel=M,
            kernel_size=K,
            activation=activation,
            name=f"act_{activation}",
        )
        assert block.gabor_activation is not None
        assert type(block.gabor_activation) is keras.layers.Activation

        y = block(sample_input, training=False)
        assert tuple(y.shape) == (B, H, W, F)
        assert bool(keras.ops.all(keras.ops.isfinite(y)))

    def test_activation_kwargs_are_dropped_on_the_fallback_path_only(self):
        """The D-012 kwargs-drop caveat is REAL, and observable numerically.

        The class docstring claims `activation_kwargs` are silently dropped
        when the activation is not a registry key. That claim is checked here
        against its own contrast: the SAME kwarg reaches the layer on the
        registry path and does not on the fallback path. A one-sided version
        of this test could not tell "dropped" from "never supported".
        """
        slope = 0.25

        # Registry path: 'relu' -> create_activation_layer -> keras.layers.ReLU,
        # which accepts `negative_slope` and keeps it.
        registry_block = GaborDepthwiseSeparableBlock(
            filters=F,
            filters_per_channel=M,
            kernel_size=K,
            activation="relu",
            activation_kwargs={"negative_slope": slope},
            name="kwargs_registry",
        )
        assert registry_block.gabor_activation.negative_slope == pytest.approx(slope)

        # Fallback path: 'leaky_relu' -> keras.layers.Activation, which takes
        # no `negative_slope` at all -- the kwarg is dropped by the resolver.
        fallback_block = GaborDepthwiseSeparableBlock(
            filters=F,
            filters_per_channel=M,
            kernel_size=K,
            activation="leaky_relu",
            activation_kwargs={"negative_slope": slope},
            name="kwargs_fallback",
        )
        act = fallback_block.gabor_activation
        assert type(act) is keras.layers.Activation
        assert not hasattr(act, "negative_slope")
        assert "negative_slope" not in act.get_config()

        # And the drop is visible in the NUMBERS, not only in the attributes:
        # Keras' own `leaky_relu` default slope is 0.2, so -2.0 maps to -0.4.
        # Had the kwarg survived, slope 0.25 would give -0.5. MEASURED: -0.4.
        probe = np.array([[-2.0, -1.0, 0.0, 1.0]], dtype="float32")
        out = keras.ops.convert_to_numpy(act(probe))
        np.testing.assert_allclose(out, [[-0.4, -0.2, 0.0, 1.0]], rtol=0, atol=1e-6)
        assert out[0, 0] != pytest.approx(-2.0 * slope), (
            "the fallback activation honoured `negative_slope`, so the "
            "docstring's kwargs-drop caveat is wrong"
        )

    def test_both_stages_on_run_in_the_dw_norm_act_pw_order(self, sample_input):
        """Stage order is depthwise -> norm -> activation -> pointwise (D-004).

        Asserted STRUCTURALLY, by recomposing the block's own sub-layers in the
        two rival orders and comparing against what `call()` actually produced
        -- not by reading the source. `layer_norm` and `relu` do not commute
        (relu clips before the norm sees the negative half), so the swapped
        order is numerically distinguishable; the separation assertion below
        proves the discriminator is live, which is what keeps the equality
        assertion from being satisfiable by any order at all.
        """
        block = GaborDepthwiseSeparableBlock(
            filters=F,
            filters_per_channel=M,
            kernel_size=K,
            normalization_type="layer_norm",
            activation="relu",
            name="both_on",
        )
        assert block.gabor_norm is not None
        assert block.gabor_activation is not None

        y = keras.ops.convert_to_numpy(block(sample_input, training=False))
        assert np.all(np.isfinite(y))

        depthwise = block.gabor_depthwise(sample_input, training=False)
        as_specified = keras.ops.convert_to_numpy(
            block.pointwise_conv(
                block.gabor_activation(block.gabor_norm(depthwise, training=False)),
                training=False,
            )
        )
        swapped = keras.ops.convert_to_numpy(
            block.pointwise_conv(
                block.gabor_norm(block.gabor_activation(depthwise), training=False),
                training=False,
            )
        )

        # MEASURED: the specified order reproduces `call()` at max abs diff
        # 0.0 (identical ops in identical order). The 1e-6 allowance is only
        # headroom against kernel-selection non-determinism, not slack in the
        # claim.
        assert np.max(np.abs(y - as_specified)) <= 1e-6

        # MEASURED separation between the two orders: 2.47 on a probe run (the
        # exact value moves with the randomly initialised pointwise kernel, so
        # the threshold is set ~20x below it rather than pinned to it).
        separation = float(np.max(np.abs(as_specified - swapped)))
        assert separation > 0.1, (
            f"norm and activation commute on this input (separation "
            f"{separation}), so the order assertion above cannot fail"
        )

        # Sub-layer declaration order is a second, independent witness.
        assert [layer.name for layer in block._layers] == [
            "gabor_depthwise",
            "gabor_norm",
            "gabor_activation",
            "pointwise_conv",
        ]


# ---------------------------------------------------------------------
# Q. Degenerate spatial inputs
# ---------------------------------------------------------------------

# The symbolic `keras.Input((None, None, C))` arm asked for by plan step 5 is
# NOT repeated here: section C's `test_symbolic_none_spatial_dims` already
# traces it, asserts the static channel count `(None, None, None, F)`, and runs
# a concrete batch through the resulting `keras.Model`. Duplicating it would
# add a test without adding a claim.

class TestDegenerateSpatialInputs:
    """What the block does when 'valid' padding runs out of pixels.

    Both behaviours below are MEASURED, then asserted. Neither was predicted:
    the block's `compute_output_shape` and its `call()` DISAGREE once the
    kernel is larger than the input, and the pair of arms pins that divergence
    where a later reader will find it rather than rediscover it.
    """

    def test_kernel_larger_than_input_raises_from_call(self):
        """K=3 on a 1x1 input with 'valid' padding: the conv itself raises.

        MEASURED message (from Keras' conv shape check, not from the block's
        own validation, which only guards rank and channel count):
            "Computed output size would be negative. Received `inputs
             shape=(1, 1, 1, 3)`, `kernel shape=(3, 3, 3, 6)` ..."
        """
        block = GaborDepthwiseSeparableBlock(
            filters=F,
            filters_per_channel=M,
            kernel_size=K,
            padding="valid",
            name="degenerate_1x1",
        )
        with pytest.raises(ValueError, match=r"[Cc]omputed output size would be negative"):
            block(np.zeros((1, 1, 1, 3), dtype="float32"), training=False)

    def test_compute_output_shape_raises_where_call_raises(self):
        """The unbuilt shape path REPRODUCES `call()`'s refusal (review W-2).

        This arm previously pinned the opposite: `compute_output_shape` used to
        answer `(1, -1, -1, 8)` -- an arithmetically consistent but physically
        impossible extent -- where `call()` raised. That divergence came from the
        block owning a second, private copy of the spatial arithmetic. Now that
        both paths delegate to the sub-layers (decisions.md D-023), the
        depthwise stage raises here exactly as it does in `call()`.

        MEASURED on keras 3.8.0, this is also what BOTH stock layers the block
        wraps already do on the same shape:
            `keras.layers.Conv2D(8, 3, padding='valid')
                 .compute_output_shape((1, 1, 1, 3))`            -> RAISES
            `keras.layers.DepthwiseConv2D(3, depth_multiplier=2,
                 padding='valid').compute_output_shape((1,1,1,3))` -> RAISES
        both with `ValueError: Computed output size would be negative`. The
        boundary one pixel over (`2x2 -> (1, 0, 0, 8)`), where the block, its
        `call()` and both stock layers all agree, is pinned by the arm below.
        """
        block = GaborDepthwiseSeparableBlock(
            filters=F,
            filters_per_channel=M,
            kernel_size=K,
            padding="valid",
            name="degenerate_cos",
        )
        assert block.built is False
        with pytest.raises(ValueError, match=r"[Cc]omputed output size would be negative"):
            block.compute_output_shape((1, 1, 1, 3))
        # The query must not have built the layer as a side effect.
        assert block.built is False

    def test_the_stock_keras_layers_agree_on_the_impossible_config(self):
        """Anti-vacuity control for the arm above: the block matches Keras.

        If a future Keras release starts CLAMPING instead of raising, this arm
        goes red first and names the framework, so the block's behaviour is not
        silently re-litigated as a defect of this layer.
        """
        for layer in (
            keras.layers.Conv2D(F, K, padding="valid"),
            keras.layers.DepthwiseConv2D(K, depth_multiplier=M, padding="valid"),
        ):
            with pytest.raises(ValueError, match=r"[Cc]omputed output size would be negative"):
                layer.compute_output_shape((1, 1, 1, 3))

    def test_exactly_zero_spatial_extent_is_produced_not_rejected(self):
        """K=3 on a 2x2 input with 'valid' padding: extent 0, and it runs.

        MEASURED: `compute_output_shape` gives `(1, 0, 0, 8)` and `call()`
        returns a tensor of that shape without raising -- the boundary case one
        pixel away from the raising arm above, where the two paths DO agree.

        Note the deliberate absence of an `isfinite` assertion: MEASURED,
        `keras.ops.all(keras.ops.isfinite(y))` is `True` on a zero-element
        tensor for the same reason `all([])` is -- vacuously. Asserting it here
        would be a guard that cannot fail, so the size is asserted instead.
        """
        block = GaborDepthwiseSeparableBlock(
            filters=F,
            filters_per_channel=M,
            kernel_size=K,
            padding="valid",
            name="degenerate_zero",
        )
        assert tuple(block.compute_output_shape((1, 2, 2, 3))) == (1, 0, 0, F)

        y = block(np.zeros((1, 2, 2, 3), dtype="float32"), training=False)
        assert tuple(y.shape) == (1, 0, 0, F)
        assert int(np.prod(y.shape)) == 0


# ---------------------------------------------------------------------
# R. POSITIVE HOMOGENEITY (SC-18) -- the layer's headline invariant
# ---------------------------------------------------------------------

# Tolerance DERIVATION (do not replace with a pasted constant).
#
# float32 carries a 24-bit significand, so its unit roundoff is
# `np.finfo(np.float32).eps` = 2**-23 = 1.1920929e-07 (MEASURED, printed by the
# probe that produced the numbers below).
#
# At defaults the block is exactly linear: a frozen depthwise dot product of
# kh*kw = K*K = 9 terms feeding a pointwise dot product of C*M = 3*2 = 6 terms,
# with `use_bias=False` on both stages and no normalization and no activation.
# Scaling the input by `a` scales every partial product by exactly `a` -- a
# float32 multiply by a common factor is exact up to one rounding -- so
# `D(a*x)` and `a*D(x)` differ ONLY in how those two accumulations round at a
# different exponent. The standard bound for a length-n float32 accumulation is
# n*eps relative to the sum of |terms|; with n <= 9 + 6 = 15 across both stages
# that is 15*eps ~= 1.8e-06 of the OUTPUT SCALE. Rounding that up to a power of
# two gives the allowance used here:
#
#     atol = 32 * eps * a * max|D(x)|      (~= 3.8e-06 of the output scale)
#     rtol = 0
#
# The tolerance is scaled to the output MAGNITUDE, not applied per element,
# and that choice is forced by measurement: individual output elements pass
# through zero, where a per-element relative error is meaningless. MEASURED on
# the exact blocks these arms construct (seed 0, `sample_input`): the
# per-element relative error reaches 1.28e-03 at a=3.7 and 2.60e-04 at a=100
# purely from near-zero elements, while the error relative to the output SCALE
# is 2.03e-07 and 1.49e-07 respectively. Across all five scales the measured
# defect is 0.00 - 1.77 units of eps of the output scale, so the 32-eps bound
# carries 18x - 23x headroom at the inexact scales:
#
#     a=0.013  defect 9.31e-09  scale 5.65e-02  atol 2.16e-07  (1.38 eps, 23.1x)
#     a=0.5    defect 0.00e+00  scale 2.17e+00  atol 8.29e-06  (0.00 eps)
#     a=2.0    defect 0.00e+00  scale 8.69e+00  atol 3.32e-05  (0.00 eps)
#     a=3.7    defect 2.86e-06  scale 1.61e+01  atol 6.13e-05  (1.49 eps, 21.4x)
#     a=100.0  defect 9.16e-05  scale 4.35e+02  atol 1.66e-03  (1.77 eps, 18.1x)
#
# a=0.5 and a=2.0 measure EXACTLY 0.0: powers of two rescale float32 without
# any rounding at all, which is why the three non-power-of-two scales are the
# ones that actually exercise the tolerance.
_F32_EPS = float(np.finfo(np.float32).eps)

# Spanning ~4 decades, deliberately mixing exact (power-of-two) and inexact
# scale factors.
_HOMOGENEITY_SCALES = (0.013, 0.5, 2.0, 3.7, 100.0)


def _homogeneity_defect(block, x, a):
    """Return ``(max|D(a*x) - a*D(x)|, max|a*D(x)|)`` -- the defect and its scale."""
    base = keras.ops.convert_to_numpy(block(x, training=False))
    got = keras.ops.convert_to_numpy(block(a * x, training=False))
    want = a * base
    return float(np.max(np.abs(got - want))), float(np.max(np.abs(want)))


def _homogeneity_atol(scale):
    """The derived allowance: 32 units of float32 eps at the output's own scale."""
    return 32.0 * _F32_EPS * scale


class TestPositiveHomogeneity:
    """`D(a*x) == a*D(x)` for `a > 0` -- plan Invariant #2, SC-18.

    This is the property D-004's bias-free defaults exist to provide and the
    one both intended consumers (`convunext`, `bfunet`, bias-free denoisers)
    depend on. Until this section existed nothing in the module measured it:
    the adversarial pass inserted `+ 0.1` after the depthwise stage inside
    `call()` -- changing no shape, no dtype, no weight path and no `.keras`
    round trip -- and the whole suite stayed green.
    """

    @pytest.mark.parametrize("a", _HOMOGENEITY_SCALES)
    def test_homogeneity_holds_at_the_bias_free_defaults(self, sample_input, a):
        """The positive arm, at the configuration D-004 ships: no norm, no
        activation, `pointwise_use_bias=False`."""
        block = GaborDepthwiseSeparableBlock(
            filters=F,
            filters_per_channel=M,
            kernel_size=K,
            pointwise_use_bias=False,
            name=f"homog_{str(a).replace('.', '_')}",
        )
        defect, scale = _homogeneity_defect(block, sample_input, a)
        # Anti-vacuity: a block whose output is identically zero would satisfy
        # any homogeneity assertion trivially. The floor is applied to the
        # UNSCALED output `max|D(x)| = scale / a`, so it means the same thing
        # at a=0.013 as at a=100.
        assert scale / a > 0.1, f"output scale {scale / a} is too small to test against"
        assert defect <= _homogeneity_atol(scale), (
            f"positive homogeneity broken at a={a}: max|D(a*x) - a*D(x)| = "
            f"{defect} exceeds {_homogeneity_atol(scale)} "
            f"(= 32 * {_F32_EPS} * {scale})"
        )

    def test_a_nonzero_pointwise_bias_breaks_homogeneity(self, sample_input):
        """THE NEGATIVE TWIN -- and it has a trap in it, MEASURED.

        `pointwise_use_bias=True` ALONE does not break homogeneity, because
        Keras initialises a `Conv2D` bias with `Zeros` (MEASURED:
        `block.pointwise_conv.bias_initializer` is a `Zeros` instance and
        `max|bias| == 0.0` on a freshly built block). Writing the twin as
        "construct with a bias, assert homogeneity fails" would therefore have
        been a guard that CANNOT FAIL -- it would have been red on arrival and
        the positive arm above would have been left unproven. MEASURED at the
        zero-initialised bias on THIS block: defect 0.0 / 0.0 / 2.86e-06 /
        6.10e-05 at a = 0.5 / 2.0 / 3.7 / 100.0, i.e. indistinguishable from
        the bias-free arm. With the bias assigned below it becomes 0.375 /
        2.025 / 74.25 at a = 0.5 / 3.7 / 100.0 -- six to nine decades larger.

        So the bias is assigned a real value first. With bias `b`,
        `D(a*x) = a*W*G(x) + b` while `a*D(x) = a*W*G(x) + a*b`, so the defect
        is exactly `|1 - a| * max|b|` -- a closed form this test checks against
        rather than merely asserting "not close".
        """
        # DECISION plan-2026-09-05T115518-e69163e4/D-026: the bias MUST be
        # assigned. Do NOT reduce this twin to "construct with
        # `pointwise_use_bias=True`, assert homogeneity fails" -- MEASURED,
        # Keras initialises a `Conv2D` bias with `Zeros`, so that version is
        # RED on arrival and leaves the positive arm above unproven. The
        # `.assign` below is what makes the twin discriminating.
        # See decisions.md D-026.
        bias_value = 0.75
        block = GaborDepthwiseSeparableBlock(
            filters=F,
            filters_per_channel=M,
            kernel_size=K,
            pointwise_use_bias=True,
            name="homog_bias",
        )
        block(sample_input, training=False)  # build

        # (i) with the shipped ZERO bias the property still holds -- this is
        #     the measurement that makes the twin non-trivial.
        assert float(np.max(np.abs(
            keras.ops.convert_to_numpy(block.pointwise_conv.bias)))) == 0.0
        defect, scale = _homogeneity_defect(block, sample_input, 3.7)
        assert defect <= _homogeneity_atol(scale), (
            "a zero-initialised bias already broke homogeneity, so the "
            "assignment below is not what makes this twin fail"
        )

        # (ii) now make the bias real, and the property must break.
        block.pointwise_conv.bias.assign(
            np.full((F,), bias_value, dtype="float32")
        )
        for a in (0.5, 3.7, 100.0):
            defect, scale = _homogeneity_defect(block, sample_input, a)
            expected = abs(1.0 - a) * bias_value
            assert defect > _homogeneity_atol(scale), (
                f"a={a}: a nonzero pointwise bias did NOT break homogeneity "
                f"(defect {defect}), so the positive arm above is vacuous"
            )
            np.testing.assert_allclose(
                defect, expected, rtol=1e-5,
                err_msg=f"the defect at a={a} is not the predicted |1-a|*|b|",
            )

    @pytest.mark.parametrize(
        "activation,homogeneous",
        [("relu", True), ("leaky_relu", True), ("gelu", False)],
    )
    def test_activation_homogeneity_matches_the_docstring_allowlist(
        self, sample_input, activation, homogeneous
    ):
        """Pins the class docstring's activation claim NUMERICALLY.

        `_POSITIVELY_HOMOGENEOUS_ACTIVATIONS` is a name allowlist and the layer
        only `logger.warning`s off it; until now nothing checked that the names
        on the list actually preserve `D(a*x) == a*D(x)` or that a name off it
        actually destroys it. `relu` and `leaky_relu` are positively homogeneous
        of degree 1 (both are `max`/scale of linear pieces through the origin);
        `gelu` is not (it has a smooth, non-scale-invariant knee).

        MEASURED on the exact blocks below, at a=3.7: `relu` defect 1.43e-06
        against scale 11.73 (1.02 eps of scale, 31x inside the allowance);
        `gelu` defect 2.27 against scale 15.05, which is 1.27e+06 eps of scale
        and 3.96e+04x OUTSIDE it. `leaky_relu` sits with `relu` at 1.08 eps.
        Neither arm is anywhere near its threshold, in either direction.
        """
        block = GaborDepthwiseSeparableBlock(
            filters=F,
            filters_per_channel=M,
            kernel_size=K,
            activation=activation,
            name=f"homog_act_{activation}",
        )
        for a in (0.5, 3.7, 100.0):
            defect, scale = _homogeneity_defect(block, sample_input, a)
            assert scale / a > 0.1
            if homogeneous:
                assert defect <= _homogeneity_atol(scale), (
                    f"{activation!r} is in the positive-homogeneity allowlist "
                    f"but broke D(a*x)==a*D(x) at a={a} (defect {defect})"
                )
            else:
                assert defect > _homogeneity_atol(scale), (
                    f"{activation!r} is NOT in the allowlist yet preserved "
                    f"D(a*x)==a*D(x) at a={a} (defect {defect}); the "
                    f"allowlist's warning would then be describing nothing"
                )

    def test_under_mixed_float16_it_holds_only_to_a_float16_bound(
        self, sample_input, mixed_float16_policy
    ):
        """SC-24. The invariant is a claim about a DTYPE REGIME, not the reals.

        Every arm above runs at `float32`, where the derived allowance is
        `32 * eps_f32 * scale`. `TestDtypePolicies` advertises `mixed_float16`
        as a supported policy, so a consumer may reasonably run the block
        there -- and under half precision the SAME property holds ~4 decades
        looser. That is ordinary fp16 rounding, not a broken layer, and the
        honest response is to assert it against the fp16 bound rather than to
        loosen the float32 one (which stays exactly where it was).

        THE BOUND IS THE SAME DERIVATION, with the eps of the COMPUTE dtype
        substituted. The derivation above bounds the defect by `n * eps` of the
        output scale for an accumulation of `n <= kh*kw + C*M = 9 + 6 = 15`
        terms, rounded up to 32. Nothing in that argument is float32-specific;
        at `mixed_float16` the block computes in float16
        (`_EXPECTED_POLICY_DTYPES`), so the same bound reads `32 * eps_f16`.
        `eps_f16 = 9.766e-04`, i.e. 8192x `eps_f32`.

        MEASURED at this configuration (M=2, K=3, F=8), worst over the five
        scales: relative defect `1.267e-03` = 1.297 units of `eps_f16` --
        24.7x inside the 32-eps bound, the same order of headroom the float32
        arms carry (1.4-1.8 eps). Per scale: a=0.013 1.11 eps_f16, a=0.5 and
        a=2.0 EXACTLY 0 (powers of two rescale without rounding in any radix-2
        format), a=3.7 1.01 eps_f16, a=100.0 1.30 eps_f16. At the SHIPPED
        defaults (M=4, K=11) the same measurement gives 4.58 eps_f16, still
        inside 32 but with less headroom -- expected, since `n` is 133 there,
        which is why this arm is pinned at the configuration the derivation
        was written for.
        """
        assert mixed_float16_policy == "mixed_float16"
        block = GaborDepthwiseSeparableBlock(
            filters=F,
            filters_per_channel=M,
            kernel_size=K,
            pointwise_use_bias=False,
            name="homog_f16",
        )
        assert block.compute_dtype == "float16", (
            "this arm is only meaningful in half precision; the block computes "
            f"in {block.compute_dtype}"
        )

        eps16 = float(np.finfo(np.float16).eps)
        worst_rel = 0.0
        worst_over_f32_allowance = 0.0
        for a in _HOMOGENEITY_SCALES:
            defect, scale = _homogeneity_defect(block, sample_input, a)
            assert scale / a > 0.1, f"output scale {scale / a} is too small"
            assert defect <= 32.0 * eps16 * scale, (
                f"positive homogeneity at a={a} broke even the FLOAT16 bound: "
                f"defect {defect} exceeds {32.0 * eps16 * scale} "
                f"(= 32 * {eps16} * {scale}); that is a layer defect, not "
                f"half-precision rounding"
            )
            worst_rel = max(worst_rel, defect / scale)
            worst_over_f32_allowance = max(
                worst_over_f32_allowance, defect / _homogeneity_atol(scale)
            )

        # ANTI-VACUITY, and the reason the docstring now carries a dtype
        # qualifier: this arm must NOT be satisfiable by the float32 allowance,
        # otherwise it would be a second float32 run wearing an fp16 label and
        # the "~4 decades looser" claim would be unevidenced.
        assert worst_over_f32_allowance > 1.0, (
            "the fp16 run met the FLOAT32 homogeneity allowance "
            f"(worst defect/atol_f32 = {worst_over_f32_allowance}), so either "
            "the policy did not take effect or this arm duplicates the float32 "
            "ones; MEASURED it is ~332x outside that allowance"
        )
        assert worst_rel > 8.0 * _F32_EPS, (
            f"the worst fp16 relative defect {worst_rel} is at float32 scale, "
            "so half precision is not actually in force"
        )


# ---------------------------------------------------------------------
# S. NO DEAD KNOBS (SC-20) -- every knob is read off the BUILT sub-layer
# ---------------------------------------------------------------------

class TestKnobsReachTheBuiltSublayers:
    """Four constructor parameters were MEASURED to be fully ignorable.

    The adversarial pass made each of `kernel_initializer`,
    `kernel_regularizer`, `pointwise_use_bias` and `normalization_kwargs`
    completely inoperative inside the layer, one at a time, and the suite
    stayed green every time. `kernel_initializer` is the pointed one: D-010's
    entire trade-off is about how that initializer is resolved and what
    `get_config()` therefore emits, yet nothing checked it ever reached the
    block's ONLY learnable weight.

    Every assertion below reads the built sub-layer or the built variable, not
    the block's own stored attribute -- an attribute assertion would pass under
    all four of those mutations.
    """

    def test_kernel_initializer_reaches_the_pointwise_kernel(self, sample_input):
        """Identity of the object AND the value it actually wrote."""
        init = keras.initializers.Constant(0.125)
        block = GaborDepthwiseSeparableBlock(
            filters=F,
            filters_per_channel=M,
            kernel_size=K,
            kernel_initializer=init,
            name="knob_init",
        )
        # `keras.initializers.get(<Initializer instance>)` is the identity, and
        # `Conv2D` stores what it is handed, so the SAME object must arrive.
        assert block.pointwise_conv.kernel_initializer is init

        block(sample_input, training=False)
        kernel = keras.ops.convert_to_numpy(block.pointwise_conv.kernel)
        np.testing.assert_allclose(kernel, 0.125, rtol=0, atol=0)

        # Negative control: the DEFAULT initializer does not produce that
        # value, so the assertion above cannot be satisfied by ignoring the knob.
        default_block = GaborDepthwiseSeparableBlock(
            filters=F, filters_per_channel=M, kernel_size=K, name="knob_init_ctl"
        )
        default_block(sample_input, training=False)
        default_kernel = keras.ops.convert_to_numpy(
            default_block.pointwise_conv.kernel
        )
        assert isinstance(
            default_block.pointwise_conv.kernel_initializer,
            keras.initializers.HeNormal,
        )
        assert float(np.max(np.abs(default_kernel - 0.125))) > 1e-3

    def test_kernel_regularizer_reaches_the_pointwise_conv(self, sample_input):
        """The regularizer must produce a real, correctly-valued loss term."""
        strength = 1e-3
        reg = keras.regularizers.L2(strength)
        block = GaborDepthwiseSeparableBlock(
            filters=F,
            filters_per_channel=M,
            kernel_size=K,
            kernel_regularizer=reg,
            name="knob_reg",
        )
        assert block.pointwise_conv.kernel_regularizer is reg

        block(sample_input, training=False)
        losses = [float(keras.ops.convert_to_numpy(v)) for v in block.losses]
        assert len(losses) == 1, f"expected exactly one regularization loss, got {losses}"

        kernel = keras.ops.convert_to_numpy(block.pointwise_conv.kernel)
        np.testing.assert_allclose(
            losses[0], strength * float(np.sum(kernel ** 2)), rtol=1e-5,
            err_msg="the loss term is not L2 of the pointwise kernel",
        )

        # Negative control at the shipped default (`kernel_regularizer=None`):
        # no loss term at all, so the length assertion above is discriminating.
        control = GaborDepthwiseSeparableBlock(
            filters=F, filters_per_channel=M, kernel_size=K, name="knob_reg_ctl"
        )
        control(sample_input, training=False)
        assert control.pointwise_conv.kernel_regularizer is None
        assert control.losses == []

    @pytest.mark.parametrize("pointwise_use_bias", [True, False])
    def test_pointwise_use_bias_controls_the_built_bias_variable(
        self, sample_input, pointwise_use_bias
    ):
        """The knob must move the sub-layer flag AND the weight layout."""
        block = GaborDepthwiseSeparableBlock(
            filters=F,
            filters_per_channel=M,
            kernel_size=K,
            pointwise_use_bias=pointwise_use_bias,
            name=f"knob_bias_{pointwise_use_bias}",
        )
        assert block.pointwise_conv.use_bias is pointwise_use_bias

        block(sample_input, training=False)
        paths = _relative_weight_paths(block)
        if pointwise_use_bias:
            assert block.pointwise_conv.bias is not None
            assert paths == EXPECTED_WEIGHT_SUFFIXES | {"pointwise_conv/bias"}
            assert len(block.weights) == 3
        else:
            assert block.pointwise_conv.bias is None
            assert paths == EXPECTED_WEIGHT_SUFFIXES
            assert len(block.weights) == 2

    def test_normalization_kwargs_reach_the_norm_sublayer(self, sample_input):
        """A non-default `epsilon` must survive the factory call.

        Step 1(b) of this plan MEASURED that `create_normalization_layer`'s own
        `epsilon` default is 1e-6 (a bare `keras.layers.LayerNormalization()`
        would be 1e-3 -- the 1000x hazard S-3 names). 1e-2 is therefore
        distinguishable from BOTH, so this arm cannot be satisfied by a dropped
        `**normalization_kwargs` falling back to either default.
        """
        block = GaborDepthwiseSeparableBlock(
            filters=F,
            filters_per_channel=M,
            kernel_size=K,
            normalization_type="layer_norm",
            normalization_kwargs={"epsilon": 1e-2},
            name="knob_norm_kwargs",
        )
        assert block.gabor_norm.epsilon == pytest.approx(1e-2)

        block(sample_input, training=False)
        assert block.gabor_norm.epsilon == pytest.approx(1e-2)

        # Negative control: the factory default, MEASURED as 1e-6.
        control = GaborDepthwiseSeparableBlock(
            filters=F,
            filters_per_channel=M,
            kernel_size=K,
            normalization_type="layer_norm",
            name="knob_norm_kwargs_ctl",
        )
        assert control.gabor_norm.epsilon == pytest.approx(1e-6)


# ---------------------------------------------------------------------
# T. The SHIPPED DEFAULTS, and a fully-ON `.keras` round trip (review W-5)
# ---------------------------------------------------------------------

# Almost every built test in this module uses `filters_per_channel=2,
# kernel_size=3` -- a configuration chosen so 'valid' padding stays
# distinguishable on a 16x16 input, but NOT the configuration the layer
# documents and ships. The two arms below cover the gap: the documented
# defaults are actually run, and a save/load carries a config with every
# optional stage switched on.

_ON_ARM_KWARGS = dict(
    filters=F,
    filters_per_channel=M,
    kernel_size=K,
    normalization_type="layer_norm",
    normalization_kwargs={"epsilon": 1e-4},
    activation="relu",
    activation_kwargs={"negative_slope": 0.1},
    pointwise_use_bias=True,
)


class TestShippedDefaultsAndOnArmRoundTrip:
    """The documented defaults run, and an ON config survives `.keras`."""

    def test_the_shipped_defaults_build_and_forward(self, sample_input):
        """`filters_per_channel=4, kernel_size=11` -- constructed elsewhere in
        this module, never built or forward-passed until here.

        `kernel_size=11` on a 16x16 input with the default `padding='same'`
        keeps the spatial extent, and `depth_multiplier=4` on 3 channels gives
        a 12-channel depthwise stage, so the layout is genuinely different from
        the (2, 3) configuration the rest of the suite rides on.
        """
        block = GaborDepthwiseSeparableBlock(filters=F, name="shipped_defaults")
        assert block.filters_per_channel == 4
        assert block.kernel_size == 11
        assert block.strides == 1
        assert block.padding == "same"

        y = block(sample_input, training=False)
        assert tuple(y.shape) == (B, H, W, F)
        assert bool(keras.ops.all(keras.ops.isfinite(y)))

        # The intermediate width the defaults imply: 3 channels x 4 = 12.
        assert tuple(block.gabor_depthwise.kernel.shape) == (11, 11, 3, 4)
        assert _relative_weight_paths(block) == EXPECTED_WEIGHT_SUFFIXES
        assert tuple(block.pointwise_conv.kernel.shape) == (1, 1, 12, F)

    def test_every_optional_stage_on_survives_the_keras_round_trip(
        self, sample_input, tmp_path
    ):
        """Norm AND activation AND regularizer AND bias, all through save/load.

        Section F's round trip covers the defaults arm only, so no ON-arm
        config -- and therefore none of `normalization_kwargs`,
        `activation_kwargs`, `kernel_regularizer` or `pointwise_use_bias` --
        was ever actually serialized and restored. Same strictness as section
        F: `rtol=0`, `atol=0`, explicit `training=False` on both calls, and the
        weight comparison taken BEFORE the loaded model's first call.
        """
        inputs = keras.Input(shape=(H, W, 3))
        block = GaborDepthwiseSeparableBlock(
            kernel_regularizer=keras.regularizers.L2(1e-4),
            name="on_blk",
            **_ON_ARM_KWARGS,
        )
        model = keras.Model(inputs, block(inputs))

        y0 = model(sample_input, training=False)
        assert bool(keras.ops.all(keras.ops.isfinite(y0)))

        path = os.path.join(tmp_path, "gabor_dsb_on.keras")
        model.save(path)
        loaded = keras.models.load_model(path)

        original = {w.path: keras.ops.convert_to_numpy(w) for w in model.weights}
        restored = {w.path: keras.ops.convert_to_numpy(w) for w in loaded.weights}
        assert set(original) == set(restored)
        # dw kernel + norm gamma/beta + pw kernel + pw bias.
        assert len(original) == 5
        for path_key in original:
            np.testing.assert_allclose(
                original[path_key], restored[path_key], rtol=0.0, atol=0.0,
                err_msg=f"weight {path_key} changed across the .keras round trip",
            )

        y1 = loaded(sample_input, training=False)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(y0),
            keras.ops.convert_to_numpy(y1),
            rtol=0.0,
            atol=0.0,
            err_msg="reloaded ON-arm model is not bit-identical to the original",
        )

        # Every ON knob must have come back as a live sub-layer property, not
        # just as a key in the config dict.
        rblock = loaded.get_layer("on_blk")
        assert rblock.gabor_norm is not None
        assert rblock.gabor_norm.epsilon == pytest.approx(1e-4)
        assert rblock.gabor_activation is not None
        assert rblock.pointwise_conv.use_bias is True
        assert rblock.pointwise_conv.bias is not None
        assert rblock.pointwise_conv.kernel_regularizer is not None
        assert len(rblock.losses) == 1
        # And the freeze still survives, on the ON arm too.
        assert rblock.gabor_depthwise.trainable is False
        assert rblock.gabor_depthwise.kernel.path not in {
            w.path for w in rblock.trainable_weights
        }


# ---------------------------------------------------------------------
# U. A TRAINING-DEPENDENT normalization stage (review W-6)
# ---------------------------------------------------------------------

class TestTrainingDependentNormalization:
    """`batch_norm` is the only norm arm here whose output depends on `training`.

    The two norms section P exercises (`layer_norm`, `rms_norm`) have no
    training-mode behaviour at all, so the block's training-mode contract was
    unmeasured for the norms that need it. `batch_norm` and
    `bias_free_batch_norm` are equally valid keys of the same factory and DO
    depend on it.

    SCOPE NOTE, MEASURED, so a later reader does not mistake this section for
    something it is not: these arms do NOT guard the EXPLICIT
    `training=training` argument on `self.gabor_norm(...)`. Keras 3.8 also
    delivers `training` to sub-layers through an AMBIENT call context, so with
    that argument deleted these three arms stay GREEN (MEASURED: the whole
    module at `96 passed`). What separates the explicit channel from the
    ambient one is a POISONER, and that guard lives in section V below.

    An earlier revision of this docstring went further and claimed the
    explicit forward was unobservable by any test at all. That claim was
    REFUTED and is deleted -- see section V and decisions.md D-027.

    What these arms DO pin, and what was genuinely missing, is that the block
    runs a training-dependent normalization correctly in BOTH modes -- an
    entire norm family (`batch_norm`, `bias_free_batch_norm`) that no arm in
    this module previously touched, since `layer_norm` and `rms_norm` have no
    training-mode behaviour at all.
    """

    # DECISION plan-2026-09-05T115518-e69163e4/D-027: this section does NOT
    # guard `training=` forwarding, and must not be relabelled as if it did --
    # MEASURED, deleting `training=training` from `self.gabor_norm(...)` leaves
    # these arms green, because Keras 3.8 also delivers the flag ambiently. Do
    # NOT, however, conclude from that (as D-026 did) that no test can see the
    # explicit forward: section V reddens on exactly that mutation using the
    # repo's `_ContextPoisoner`. See decisions.md D-027.
    @pytest.mark.parametrize(
        "normalization_type", ["batch_norm", "bias_free_batch_norm"]
    )
    def test_train_and_inference_outputs_differ(self, sample_input, normalization_type):
        block = GaborDepthwiseSeparableBlock(
            filters=F,
            filters_per_channel=M,
            kernel_size=K,
            normalization_type=normalization_type,
            name=f"train_dep_{normalization_type}",
        )
        y_train = keras.ops.convert_to_numpy(block(sample_input, training=True))
        y_infer = keras.ops.convert_to_numpy(block(sample_input, training=False))
        assert np.all(np.isfinite(y_train)) and np.all(np.isfinite(y_infer))
        assert tuple(y_train.shape) == (B, H, W, F)

        # MEASURED: no warm-up is needed. On the FIRST call the moving
        # statistics are still at their initial (mean 0, var 1) values while
        # `training=True` uses the batch's own statistics, and the gap is
        # already large -- 1.11 for `batch_norm`, 1.68 for `bias_free_batch_norm`
        # against outputs of order 1. The threshold is set two decades below
        # the smaller of those, not pinned to either.
        first_call_gap = float(np.max(np.abs(y_train - y_infer)))
        assert first_call_gap > 1e-2, (
            f"{normalization_type} produced the same output in both modes "
            f"(gap {first_call_gap}), so this arm measures nothing"
        )

        # And it stays true after the moving statistics have actually moved,
        # so the claim is not an artefact of the untouched initial state.
        # MEASURED after 20 training calls: 0.968 / 1.46 respectively.
        for _ in range(20):
            block(sample_input, training=True)
        warm_gap = float(np.max(np.abs(
            keras.ops.convert_to_numpy(block(sample_input, training=True))
            - keras.ops.convert_to_numpy(block(sample_input, training=False))
        )))
        assert warm_gap > 1e-2, (
            f"{normalization_type} train/infer gap collapsed to {warm_gap} "
            f"once the moving statistics warmed up"
        )

    def test_the_moving_statistics_actually_move_in_training_only(self, sample_input):
        """Anti-vacuity control for the arm above: `training` reaches the norm.

        If the gap above came from something other than the training flag, the
        moving statistics would be indifferent to it. MEASURED: they advance
        under `training=True` and are bit-identical under `training=False`.

        Note what this does and does not establish. It proves the flag REACHES
        the norm sub-layer; it does not attribute that to the explicit
        `training=` argument in `call()`, because (see the class docstring)
        Keras 3.8 would deliver the flag through the ambient call context even
        if that argument were deleted. Section V does make that attribution,
        by poisoning the ambient channel first.
        """
        block = GaborDepthwiseSeparableBlock(
            filters=F,
            filters_per_channel=M,
            kernel_size=K,
            normalization_type="batch_norm",
            name="moving_stats",
        )
        block(sample_input, training=False)
        before = keras.ops.convert_to_numpy(block.gabor_norm.moving_mean).copy()

        for _ in range(5):
            block(sample_input, training=False)
        assert np.array_equal(
            keras.ops.convert_to_numpy(block.gabor_norm.moving_mean), before
        ), "inference calls moved the moving statistics"

        block(sample_input, training=True)
        after = keras.ops.convert_to_numpy(block.gabor_norm.moving_mean)
        assert not np.array_equal(after, before), (
            "a training call did NOT move the moving statistics"
        )


# ---------------------------------------------------------------------
# V. `training=` IS FORWARDED EXPLICITLY (SC-22, SC-23) -- read this first
# ---------------------------------------------------------------------
#
# WHAT THIS SECTION EXISTS TO CORRECT. An earlier revision of this module
# asserted, in section U's class docstring and in a `# DECISION .../D-026`
# anchor, that dropping `training=training` from `self.gabor_norm(...)` was
# behaviourally unobservable BY ANY TEST ANYWHERE, and instructed maintainers
# not to try to write one. That was FALSE. It was refuted with a pattern that
# already lives in this repo, and both the claim and the anchor are deleted
# (decisions.md D-027 carries the retracted wording verbatim, so this file
# does not).
#
# THE MECHANISM. Keras delivers `training` to a sub-layer through TWO channels:
#   (1) the EXPLICIT `training=` kwarg at the call site, and
#   (2) an AMBIENT `CallContext.training`.
# `keras/src/layers/layer.py:851` writes `call_context.training = training` on
# every nested `__call__`, and `_maybe_reset_call_context` (:1501) clears it
# only for the OUTERMOST entry layer -- so the ambient value is a SINGLE
# MUTABLE SLOT that any sibling sub-layer can overwrite for every LATER
# un-forwarded call in the same outer call.
#
# WHETHER A PLAIN TRAIN-VS-INFERENCE PROBE CAN SEE THE EXPLICIT FORWARD DEPENDS
# ENTIRELY ON THE ENTRY PROTOCOL, and the two protocols give OPPOSITE answers.
# RE-DERIVED at this site (fresh `batch_norm` block, `filters=8`,
# `filters_per_channel=2`, `kernel_size=3`, built by an inference call, then
# `max|out(training=True) - out(training=False)|`), at HEAD and with
# `training=training` dropped from `self.gabor_norm(...)`:
#
#   * P1 -- the PUBLIC path, `block(x, training=...)`, i.e. `Layer.__call__`:
#     `1.269153` at HEAD, `1.269153` mutated. INSENSITIVE. The outer
#     `__call__` has ALREADY written `call_context.training`
#     (`keras/src/layers/layer.py:851`) before `call()` runs, so the norm's own
#     `__call__` reads the right value off the ambient slot whether or not the
#     explicit kwarg is present.
#   * P2 -- the DIRECT path, `block.call(x, training=...)`, which BYPASSES
#     `Layer.__call__`: `1.269153` at HEAD, `0.000000` mutated. SENSITIVE. No
#     outer `__call__` runs, so nothing writes the ambient slot; the explicit
#     kwarg is then the ONLY channel, and deleting it leaves the norm at its
#     `training=None` inference default in both arms.
#
# So a direct-`.call()` arm DOES redden on this mutation. It is deliberately
# NOT shipped here, and the reason is a judgement about instruments, not an
# impossibility: `.call()` is not how any consumer reaches this block, so such
# an arm would pin the framework's kwarg plumbing on a non-public entry path
# (no build, no autocast, no name scope, no ambient slot) rather than the
# block's behaviour on the path callers actually use. A later maintainer who
# wants it anyway as a second mechanism is not doing something futile -- see
# the DECISION anchor below. An earlier revision of this header asserted that
# such a comparison "CANNOT redden ... it never will"; that assertion was
# FALSE, and the numbers above are its retraction (decisions.md D-030,
# corrected).
#
# The `_ContextPoisoner` below is the SHIPPED instrument precisely because it
# reddens on the PUBLIC path P1 -- it recovers the observability that P1 loses
# to the ambient slot, while still isolating the explicit kwarg. It separates
# the channels by delegating to a real sub-layer and then calling a dead child
# with `training=False`, poisoning the ambient slot mid-call. It is IMPORTED, not re-implemented: it is this
# repo's established, documented, PROVEN-RED instrument for this exact defect
# class (see the block comment at
# `tests/test_layers/test_attention/test_tripse_attention.py:627-645`, which
# records it firing on `'gate_activation' received training=False` before that
# plan's D-015 fix). This module already imports a shared test oracle across
# packages the same way (`tests.test_models.gradient_flow_oracle`).
#
# MEASURED on THIS layer, poisoner installed at `gabor_depthwise`, block called
# `training=True`:
#   * `gabor_norm.moving_mean` movement: 0.00008846 at HEAD, 0.00000000 with
#     `training=training` dropped from `self.gabor_norm(...)`; the UNPOISONED
#     control moves 0.00008846 in both cases -- so it is the poisoner, not the
#     probe, that makes the gap observable.
#   * the activation observed `training=[False]` before it was given an
#     explicit forward, and `[True]` after (review pass-2 R-2, decisions.md
#     D-028).
#
# DECISION plan-2026-09-05T115518-e69163e4/D-027 + D-029 + D-032: this section
# is the guard the deleted anchor said could not exist. Do NOT weaken it by
# dropping the poisoner -- without it every arm here passes on the ambient channel and the
# section becomes vacuous, which is precisely how the false claim arose. Do
# NOT replace the poisoner with a local copy either: one definition, in the
# module whose header carries its proven-RED provenance. A direct
# `block.call(x, training=True/False)` arm is NOT shipped as a second
# mechanism, but that is a CHOICE, not an impossibility: it does redden on the
# mutation (MEASURED `1.269153` at HEAD -> `0.000000` mutated; see the P1/P2
# table above). It is omitted because it enters through a non-public path that
# no consumer uses. Adding it later is legitimate -- if you do, label it as
# testing the explicit-kwarg channel in isolation and prove it RED first. Do
# NOT re-introduce the retracted claim that such an arm cannot redden. See
# decisions.md D-027, D-028, D-029, D-030 and D-032.

# Registry key for the strict probe activation, namespaced to this module so it
# cannot collide with the tripse module's own key if both are collected.
_STRICT_ACT_KEY = "__gabor_block_strict_training_probe__"


class _ShapedStrictTrainingActivation(StrictTrainingActivation):
    """`StrictTrainingActivation` plus the one method THIS block requires.

    MEASURED: `GaborDepthwiseSeparableBlock.build()` calls
    `self.gabor_activation.compute_output_shape(...)` to thread the stage shape
    (that delegation is what SC-19 rests on), and `keras.layers.Layer` has no
    default implementation -- the bare probe raises
    `NotImplementedError: Layer StrictTrainingActivation does not have a
    compute_output_shape method implemented`. TripSE never hits this because it
    does not size anything off its activation.

    Subclassing rather than copying keeps ONE definition of the probe's actual
    behaviour (the `training is not True` raise and its message) in the repo;
    only the shape passthrough is added here. Do NOT reimplement the `call()`.
    """

    def compute_output_shape(self, input_shape):
        return input_shape


@pytest.fixture
def strict_activation_registered():
    """Register `_ShapedStrictTrainingActivation` under `_STRICT_ACT_KEY`.

    Installing it in `ACTIVATION_REGISTRY` is what lets the probe enter through
    the block's own `activation=` CONSTRUCTOR argument rather than by patching
    a built attribute -- so the arm exercises the real construction path.

    :yield: the registry key the block should be constructed with.
    :rtype: str
    """
    _act_factory.ACTIVATION_REGISTRY[_STRICT_ACT_KEY] = {
        "class": _ShapedStrictTrainingActivation,
        "description": "test-only strict training probe",
        "required_params": [],
        "optional_params": {},
        "use_case": "proven-RED injection for training= forwarding",
    }
    try:
        yield _STRICT_ACT_KEY
    finally:
        _act_factory.ACTIVATION_REGISTRY.pop(_STRICT_ACT_KEY, None)


class TestTrainingIsForwardedExplicitly:
    """Read the block comment above before touching these arms."""

    @staticmethod
    def _batch_norm_block(name, poisoned):
        block = GaborDepthwiseSeparableBlock(
            filters=F,
            filters_per_channel=M,
            kernel_size=K,
            normalization_type="batch_norm",
            name=name,
        )
        if poisoned:
            # The depthwise stage is the one sub-layer that runs BETWEEN the
            # block's entry and the norm, so it is where the poison has to go.
            block.gabor_depthwise = _ContextPoisoner(block.gabor_depthwise)
        return block

    @staticmethod
    def _moving_mean_movement(block, sample_input, training):
        block(sample_input, training=False)  # build, and settle the statistics
        before = keras.ops.convert_to_numpy(block.gabor_norm.moving_mean).copy()
        block(sample_input, training=training)
        after = keras.ops.convert_to_numpy(block.gabor_norm.moving_mean)
        return float(np.max(np.abs(after - before)))

    def test_the_norm_gets_training_through_the_explicit_kwarg(self, sample_input):
        """RED when `self.gabor_norm(x, training=training)` loses its kwarg.

        MEASURED: poisoned movement 0.00008846 at HEAD and 0.00000000 with the
        forward dropped, with the unpoisoned control at 0.00008846 both ways.
        """
        poisoned = self._moving_mean_movement(
            self._batch_norm_block("poisoned_norm", poisoned=True),
            sample_input,
            training=True,
        )
        control = self._moving_mean_movement(
            self._batch_norm_block("control_norm", poisoned=False),
            sample_input,
            training=True,
        )

        # The CONTROL is what proves the poisoner (not the probe) is what makes
        # the gap visible: it passes on fixed AND unfixed code, by design.
        assert control > 0.0, (
            "the unpoisoned control did not move the moving statistics at all, "
            "so this arm's instrument is broken, not the layer"
        )
        assert poisoned == pytest.approx(control, rel=1e-6), (
            f"with the ambient CallContext poisoned the norm moved by "
            f"{poisoned} instead of the control's {control}: "
            f"`self.gabor_norm(x, training=training)` is not passing `training` "
            f"EXPLICITLY, so the norm ran in inference mode inside a "
            f"`training=True` call"
        )

    def test_the_poisoned_arm_still_sees_inference_mode_as_inference(
        self, sample_input
    ):
        """ANTI-VACUITY twin: the arm above must not pass for a trivial reason.

        If the poisoned block moved its statistics under `training=False` too,
        the assertion above would be measuring an unconditional update rather
        than a forwarded flag.
        """
        movement = self._moving_mean_movement(
            self._batch_norm_block("poisoned_infer", poisoned=True),
            sample_input,
            training=False,
        )
        assert movement == 0.0, (
            f"an inference call moved the moving statistics by {movement}"
        )

    @staticmethod
    def _ambient_name_prefix(name, sample_input):
        """Whatever Keras' global name-scope stack prepends to a fresh block.

        `''` in a clean process. NOT asserted to be empty: another module may
        legitimately have leaked a scope before this one ran (D-018), and a
        test that reads the ABSOLUTE path would then be RED purely from
        collection order -- the very defect C-3 closed in this module.
        """
        block = GaborDepthwiseSeparableBlock(
            filters=F, filters_per_channel=M, kernel_size=K, name=name
        )
        block(sample_input, training=False)
        marker = name + "/"
        paths = {w.path for w in block.weights}
        assert paths, "the probe block created no weights"
        prefixes = {p[: p.index(marker)] for p in paths if marker in p}
        assert len(prefixes) == 1, f"inconsistent weight-path prefixes: {paths}"
        return prefixes.pop()

    def test_the_poisoner_does_not_leak_a_name_scope(self, sample_input):
        """The poisoner must not become a second instance of D-018.

        `test_tripse_attention.py`'s own poisoned nesting leaves `'outer'` on
        Keras' process-global `name_scope_stack` FOREVER, which reddens weight
        path assertions in six other packages (decisions.md D-018). MEASURED,
        the shape used here does NOT leak -- this arm pins that, so a future
        change to the poisoner cannot export the leak from this module.

        It compares the ambient prefix BEFORE and AFTER, never against `''`:
        this module already shipped five order-coupled tests once (C-3), and
        an absolute-path version of this assertion would be RED whenever the
        tripse arm above ran first -- MEASURED, `1 failed, 103 passed` for
        exactly that pairing.
        """
        before = self._ambient_name_prefix("leak_probe_before", sample_input)
        self._moving_mean_movement(
            self._batch_norm_block("leak_check", poisoned=True),
            sample_input,
            training=True,
        )
        after = self._ambient_name_prefix("leak_probe_after", sample_input)
        assert after == before, (
            f"a name scope leaked out of the poisoned call: fresh blocks were "
            f"prefixed {before!r} before it and {after!r} after it, so every "
            f"variable created later in this process now carries a stray prefix"
        )

    def test_the_activation_gets_training_through_the_explicit_kwarg(
        self, sample_input, strict_activation_registered
    ):
        """SC-23. RED when `self.gabor_activation(x)` loses its `training=`.

        MEASURED before the fix: with the poisoner installed the activation
        observed `training=[False]` while the block was called `training=True`.
        `StrictTrainingActivation` is numerically the identity, so it cannot
        change the output -- the only thing it can do is fire.
        """
        block = GaborDepthwiseSeparableBlock(
            filters=F,
            filters_per_channel=M,
            kernel_size=K,
            activation=strict_activation_registered,
            name="poisoned_act",
        )
        block.gabor_depthwise = _ContextPoisoner(block.gabor_depthwise)

        out = block(sample_input, training=True)
        assert tuple(out.shape) == (B, H, W, F)

    def test_the_strict_activation_is_actually_on_the_forward_path(
        self, sample_input, strict_activation_registered
    ):
        """ANTI-VACUITY twin: the probe must be reachable at all.

        Called with `training=False` it MUST fire. Without this, an activation
        that the block silently never invoked would satisfy the arm above.
        """
        block = GaborDepthwiseSeparableBlock(
            filters=F,
            filters_per_channel=M,
            kernel_size=K,
            activation=strict_activation_registered,
            name="reachable_act",
        )
        with pytest.raises(AssertionError, match=r"dead-component injection FIRED"):
            block(sample_input, training=False)

    def test_without_the_poisoner_the_ambient_channel_already_delivers(
        self, sample_input, strict_activation_registered
    ):
        """The CONTROL that names what these arms could NOT have seen.

        Un-poisoned, an un-forwarded sub-layer still receives the right value
        through `CallContext`. This passes on fixed AND unfixed code, by
        design: it is here so nobody mistakes the arms above for evidence that
        Keras fails to propagate `training`, and so a future Keras that stops
        propagating it ambiently is noticed here rather than as a mystery.
        """
        block = GaborDepthwiseSeparableBlock(
            filters=F,
            filters_per_channel=M,
            kernel_size=K,
            activation=strict_activation_registered,
            name="unpoisoned_act",
        )
        out = block(sample_input, training=True)
        assert tuple(out.shape) == (B, H, W, F)


# ---------------------------------------------------------------------
# W. A CALLABLE `activation` SURVIVES SERIALIZATION (D-033)
# ---------------------------------------------------------------------
#
# WHY THIS SECTION EXISTS. `activation` is hinted `Optional[str]`, but Python
# does not enforce annotations and this block never coerced the value:
# `resolve_activation_layer` falls through to `keras.layers.Activation`, which
# accepts a callable, so a callable reached the forward path intact and was
# then stored RAW in `get_config()`. That is defect family N-1
# (`tests/test_the_raw_activation_config_population_is_closed.py`, which was
# RED on this exact site), and the repair is D-400's symmetric pair --
# `deserialize_activation` in `__init__`, `serialize_activation` in
# `get_config`.
#
# PROVEN RED at pristine HEAD `a9c7b300b`, in a `git worktree`, with the
# registered callable below:
#   * `json.dumps(get_config()['activation'])` -> `TypeError: Object of type
#     function is not JSON serializable`;
#   * `model.save(...)` succeeded and `keras.models.load_model(...)` then raised
#     `TypeError: <class 'keras.src.models.functional.Functional'> could not be
#     deserialized properly`.
# Both assertions below are those two failures. After the repair the config is
# a `{'class_name': 'function', 'config': 'probe>...'}` dict, the reload needs
# no `custom_objects`, and the forward output is bit-identical (max|delta| 0.0,
# `training=False` explicit on both arms).
#
# DO NOT write this arm with a REGISTERED callable ONLY and conclude the family
# is closed: D-400 measured that a registered callable round-trips on FORWARD
# OUTPUT with and without the repair on a bare-layer path. What discriminates
# is (a) `get_config()` being JSON-serializable and (b) `.activation` still
# being callable -- not `max|delta|`. Both are asserted here for that reason.


@keras.saving.register_keras_serializable(package="gabor_dsb_test")
def _registered_scaled_relu(x):
    """A REGISTERED callable activation, not a string, not a registry key.

    Scaled by 2.0 rather than being a bare `relu` so that a silent fallback to
    a default/identity activation cannot produce the same forward output.
    """
    return keras.ops.relu(x) * 2.0


def _unregistered_scaled_relu(x):
    """The same function with NO registration -- resolvable only via
    `custom_objects`. This is the value that raised
    'Could not interpret activation function identifier' in D-400's table."""
    return keras.ops.relu(x) * 2.0


class TestCallableActivationSerialization:
    """Read the block comment above before touching these arms."""

    @staticmethod
    def _model(activation):
        inputs = keras.Input(shape=(H, W, 3))
        outputs = GaborDepthwiseSeparableBlock(
            filters=F,
            filters_per_channel=M,
            kernel_size=K,
            activation=activation,
            name="blk",
        )(inputs)
        return keras.Model(inputs, outputs)

    def test_get_config_is_json_serializable_for_a_callable_activation(self):
        """Half one of the pair: `serialize_activation` in `get_config`."""
        model = self._model(_registered_scaled_relu)
        entry = model.get_layer("blk").get_config()["activation"]
        # RED pre-fix: `TypeError: Object of type function is not JSON
        # serializable`.
        json.dumps(entry)
        # Anti-vacuity: a repair that dropped the value entirely would also
        # serialize. The serialized form must still name the function.
        assert isinstance(entry, dict)
        assert "gabor_dsb_test>_registered_scaled_relu" in json.dumps(entry)

    def test_a_registered_callable_activation_survives_a_keras_round_trip(
            self, sample_input, tmp_path
    ):
        """Half two: `deserialize_activation` in `__init__`.

        No `custom_objects` is passed -- registration is what makes that
        legitimate here, and it is also what makes the arm reach the real
        `load_model` resolution path rather than a scope the test installed.
        """
        model = self._model(_registered_scaled_relu)
        y0 = model(sample_input, training=False)

        path = os.path.join(tmp_path, "gabor_callable_act.keras")
        model.save(path)
        loaded = keras.models.load_model(path)  # RED pre-fix: TypeError

        block = loaded.get_layer("blk")
        # NOT a raw dict: that is the state `serialize_activation` alone leaves
        # behind, and the next `get_config()` would propagate it onward.
        assert callable(block.activation), (
            f"activation came back as {type(block.activation).__name__}, not a "
            "callable -- deserialize_activation is not running in __init__"
        )
        assert not isinstance(block.activation, dict)
        json.dumps(block.get_config()["activation"])

        y1 = loaded(sample_input, training=False)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(y0),
            keras.ops.convert_to_numpy(y1),
            rtol=0.0,
            atol=0.0,
            err_msg="callable activation changed the forward output across a "
                    "`.keras` round trip",
        )

    def test_an_unregistered_callable_needs_only_custom_objects(
            self, sample_input, tmp_path
    ):
        """The `custom_objects` row of D-400's table.

        An UNREGISTERED callable cannot be resolved from a name by any repair --
        nothing in the file names the object. What the repair buys is that
        supplying `custom_objects` now yields a LIVE callable instead of a raw
        dict, and that `save()` no longer emits a non-JSON config.
        """
        model = self._model(_unregistered_scaled_relu)
        y0 = model(sample_input, training=False)

        path = os.path.join(tmp_path, "gabor_unregistered_act.keras")
        model.save(path)
        loaded = keras.models.load_model(
            path,
            custom_objects={"_unregistered_scaled_relu": _unregistered_scaled_relu},
        )

        block = loaded.get_layer("blk")
        assert callable(block.activation)
        assert not isinstance(block.activation, dict)

        y1 = loaded(sample_input, training=False)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(y0),
            keras.ops.convert_to_numpy(y1),
            rtol=0.0,
            atol=0.0,
        )

    def test_a_string_activation_is_untouched_by_the_pair(self):
        """The CONTROL. `serialize_activation` must pass strings through
        verbatim: this repo stores activation-factory keys such as `'mish'`
        that `keras.activations.serialize` outright REJECTS, and every shipped
        config passes a string, so the pair has to be a no-op on them."""
        for name in ("relu", "linear", "leaky_relu"):
            block = GaborDepthwiseSeparableBlock(
                filters=F, filters_per_channel=M, kernel_size=K, activation=name
            )
            assert block.activation == name
            assert block.get_config()["activation"] == name

        default = GaborDepthwiseSeparableBlock(
            filters=F, filters_per_channel=M, kernel_size=K
        )
        assert default.activation is None
        assert default.get_config()["activation"] is None
        assert default.gabor_activation is None
