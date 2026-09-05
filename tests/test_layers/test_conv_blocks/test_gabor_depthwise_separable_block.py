"""Battery for GaborDepthwiseSeparableBlock.

Sections A-F are the core arms (validation, forward, shapes, layout,
serialization). Sections G-L are the identity / freeze / gradient guards --
the ones that make this a *Gabor*, *frozen* layer -- plus the homogeneity-set
drift guard and the ``call()`` rank re-check. Sections N-Q are the robustness
arms: dtype policies, XLA-vs-eager, the optional stages switched ON (the twins
of section D's OFF assertions), and degenerate spatial inputs.
"""

import os
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

    def test_compute_output_shape_returns_a_negative_extent_where_call_raises(self):
        """The unbuilt shape helper does NOT reproduce `call()`'s refusal.

        MEASURED: for K=3, 'valid', a 1x1 input, `compute_output_shape` returns
        `(1, -1, -1, 8)` -- an arithmetically consistent but physically
        impossible extent -- while `call()` on the same shape raises (arm
        above). This is asserted as the REAL behaviour, not as the desired one:
        the layer is not modified by this step, and a caller who trusts
        `compute_output_shape` alone on an over-cropped configuration gets a
        negative dimension rather than an error.
        """
        block = GaborDepthwiseSeparableBlock(
            filters=F,
            filters_per_channel=M,
            kernel_size=K,
            padding="valid",
            name="degenerate_cos",
        )
        assert block.built is False
        assert tuple(block.compute_output_shape((1, 1, 1, 3))) == (1, -1, -1, F)

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
