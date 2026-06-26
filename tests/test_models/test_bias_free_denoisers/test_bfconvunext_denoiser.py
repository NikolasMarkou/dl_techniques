"""
Test suite for the bias-free ConvUNext denoiser (models/bias_free_denoisers/bfconvunext.py).

Covers the embedded ConvUNextStem layer (construction / forward / shape / config
round-trip), the create_convunext_denoiser functional builder (ValueError paths,
forward pass, deep supervision), the create_convunext_variant wrapper, and the
M2 full .keras save -> load -> identical-output round-trip.

NOTE: this ConvUNextStem is the *bias-free* variant (use_bias is always False),
distinct from the one in models/convunext/model.py.
"""

import os
import keras
import pytest
import numpy as np
from typing import Dict, Any, Tuple

from dl_techniques.models.bias_free_denoisers.bfconvunext import (
    ConvUNextStem,
    create_convunext_denoiser,
    create_convunext_variant,
    CONVUNEXT_CONFIGS,
)
from dl_techniques.layers.convnext_v1_block import ConvNextV1Block
from dl_techniques.layers.convnext_v2_block import ConvNextV2Block


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture
def stem_config() -> Dict[str, Any]:
    return {
        'filters': 16,
        'kernel_size': 7,
        'kernel_initializer': 'he_normal',
        'kernel_regularizer': None,
    }


@pytest.fixture
def sample_input() -> np.ndarray:
    return np.random.randn(2, 32, 32, 3).astype(np.float32)


@pytest.fixture
def input_shape() -> Tuple[int, int, int]:
    return (32, 32, 1)


# ---------------------------------------------------------------------
# ConvUNextStem (embedded layer)
# ---------------------------------------------------------------------

class TestConvUNextStem:
    """Test suite for the bias-free ConvUNextStem layer."""

    def test_instantiation(self, stem_config: Dict[str, Any]) -> None:
        stem = ConvUNextStem(**stem_config)
        assert stem.filters == stem_config['filters']
        assert stem.kernel_size == stem_config['kernel_size']

    def test_forward_pass(self, stem_config, sample_input) -> None:
        stem = ConvUNextStem(**stem_config)
        output = stem(sample_input)
        expected = (
            sample_input.shape[0],
            sample_input.shape[1],
            sample_input.shape[2],
            stem_config['filters'],
        )
        assert output.shape == expected

    def test_bias_free(self, stem_config, sample_input) -> None:
        """Stem must be bias-free: the conv uses no bias weights."""
        stem = ConvUNextStem(**stem_config)
        stem(sample_input)
        assert stem.conv.use_bias is False

    def test_build_creates_weights(self, stem_config, sample_input) -> None:
        stem = ConvUNextStem(**stem_config)
        stem(sample_input)
        assert stem.built
        assert len(stem.trainable_weights) > 0

    def test_compute_output_shape(self, stem_config, sample_input) -> None:
        stem = ConvUNextStem(**stem_config)
        computed = stem.compute_output_shape(sample_input.shape)
        actual = stem(sample_input)
        assert tuple(computed) == tuple(actual.shape)

    def test_compute_output_shape_dynamic(self, stem_config) -> None:
        stem = ConvUNextStem(**stem_config)
        computed = stem.compute_output_shape((None, 64, 64, 3))
        assert tuple(computed) == (None, 64, 64, stem_config['filters'])

    def test_config_round_trip(self, stem_config, sample_input) -> None:
        original = ConvUNextStem(**stem_config)
        original(sample_input)
        config = original.get_config()
        reconstructed = ConvUNextStem.from_config(config)
        assert reconstructed.filters == original.filters
        assert reconstructed.kernel_size == original.kernel_size


# ---------------------------------------------------------------------
# create_convunext_denoiser (functional builder)
# ---------------------------------------------------------------------

class TestCreateConvUNextDenoiser:
    """Test suite for the create_convunext_denoiser builder."""

    def _build(self, input_shape, **overrides) -> keras.Model:
        cfg = dict(
            input_shape=input_shape,
            depth=3,
            initial_filters=8,
            blocks_per_level=1,
            convnext_version='v2',
            drop_path_rate=0.0,
        )
        cfg.update(overrides)
        return create_convunext_denoiser(**cfg)

    def test_forward_pass_single_output(self, input_shape) -> None:
        model = self._build(input_shape)
        x = np.random.rand(2, *input_shape).astype(np.float32)
        y = model(x)
        assert y.shape == (2, *input_shape)

    def test_deep_supervision_multi_output(self, input_shape) -> None:
        model = self._build(input_shape, enable_deep_supervision=True)
        x = np.random.rand(1, *input_shape).astype(np.float32)
        outputs = model(x)
        assert isinstance(outputs, list)
        assert len(outputs) > 1
        # primary output keeps full resolution and input channels
        assert outputs[0].shape == (1, *input_shape)

    def test_v1_blocks(self, input_shape) -> None:
        model = self._build(input_shape, convnext_version='v1')
        x = np.random.rand(1, *input_shape).astype(np.float32)
        y = model(x)
        assert not np.any(np.isnan(keras.ops.convert_to_numpy(y)))

    # --- ValueError / TypeError paths ---

    def test_invalid_input_shape_type(self) -> None:
        with pytest.raises(TypeError, match="input_shape"):
            create_convunext_denoiser(input_shape=[32, 32, 3])

    def test_invalid_depth(self, input_shape) -> None:
        with pytest.raises(ValueError, match="depth"):
            create_convunext_denoiser(input_shape=input_shape, depth=2)

    def test_invalid_initial_filters(self, input_shape) -> None:
        with pytest.raises(ValueError, match="initial_filters"):
            create_convunext_denoiser(input_shape=input_shape, initial_filters=0)

    def test_invalid_filter_multiplier(self, input_shape) -> None:
        with pytest.raises(ValueError, match="filter_multiplier"):
            create_convunext_denoiser(input_shape=input_shape, filter_multiplier=0)

    def test_invalid_blocks_per_level(self, input_shape) -> None:
        with pytest.raises(ValueError, match="blocks_per_level"):
            create_convunext_denoiser(input_shape=input_shape, blocks_per_level=0)

    def test_invalid_convnext_version(self, input_shape) -> None:
        with pytest.raises(ValueError, match="convnext_version"):
            create_convunext_denoiser(input_shape=input_shape, convnext_version='v3')

    # --- M2 round-trip ---

    def test_keras_round_trip(self, tmp_path, input_shape) -> None:
        model = self._build(input_shape)
        x = np.random.rand(2, *input_shape).astype(np.float32)
        y_before = model(x)

        save_path = os.path.join(str(tmp_path), 'bfconvunext.keras')
        model.save(save_path)
        loaded = keras.models.load_model(save_path)
        y_after = loaded(x)

        # GPU fp32 reduction noise -> atol 1e-4 (SYSTEM invariant)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(y_before),
            keras.ops.convert_to_numpy(y_after),
            atol=1e-4,
            err_msg="Outputs differ after .keras round-trip"
        )


# ---------------------------------------------------------------------
# Depthwise init/regularizer factory pass-through (SC5 / F3)
# ---------------------------------------------------------------------

def _find_convnext_blocks(model: keras.Model):
    """Return all ConvNextV1Block/ConvNextV2Block instances in the model."""
    blocks = []
    stack = list(model.layers)
    while stack:
        layer = stack.pop()
        if isinstance(layer, (ConvNextV1Block, ConvNextV2Block)):
            blocks.append(layer)
        # recurse into nested layer-bearing layers (defensive; the factory is functional)
        nested = getattr(layer, 'layers', None)
        if nested:
            stack.extend(nested)
    return blocks


class TestDepthwisePassThrough:
    """SC5: depthwise_initializer/regularizer thread from the factory into every block.

    Also guards F3: the default factory build must NOT inject a non-None depthwise
    init/reg (the orthonormal knob must leave the default model byte-identical).
    """

    def _build(self, input_shape, version, **overrides) -> keras.Model:
        cfg = dict(
            input_shape=input_shape,
            depth=3,
            initial_filters=8,
            blocks_per_level=1,
            convnext_version=version,
            drop_path_rate=0.0,
        )
        cfg.update(overrides)
        return create_convunext_denoiser(**cfg)

    @pytest.mark.parametrize("version", ["v2", "v1"])
    def test_override_reaches_blocks(self, input_shape, version) -> None:
        init = keras.initializers.Orthogonal(gain=1.0)
        reg = keras.regularizers.L2(1e-4)
        model = self._build(
            input_shape, version,
            depthwise_initializer=init,
            depthwise_regularizer=reg,
        )

        # builds + forward-passes
        x = np.random.rand(2, *input_shape).astype(np.float32)
        y = model(x)
        assert y.shape == (2, *input_shape)
        assert not np.any(np.isnan(keras.ops.convert_to_numpy(y)))

        # the override actually reached the blocks (proves no call-site was missed)
        blocks = _find_convnext_blocks(model)
        assert len(blocks) > 0, "no ConvNeXt blocks found in the model"
        for blk in blocks:
            assert isinstance(blk.depthwise_initializer, keras.initializers.Orthogonal)
            assert isinstance(blk.depthwise_regularizer, keras.regularizers.L2)

    @pytest.mark.parametrize("version", ["v2", "v1"])
    def test_default_unchanged(self, input_shape, version) -> None:
        # F3: the factory default must NOT inject a non-None depthwise init/reg.
        model = self._build(input_shape, version)
        blocks = _find_convnext_blocks(model)
        assert len(blocks) > 0, "no ConvNeXt blocks found in the model"
        for blk in blocks:
            assert blk.depthwise_initializer is None
            assert blk.depthwise_regularizer is None

    def test_override_keras_round_trip(self, tmp_path, input_shape) -> None:
        init = keras.initializers.Orthogonal(gain=1.0)
        reg = keras.regularizers.L2(1e-4)
        model = self._build(
            input_shape, "v2",
            depthwise_initializer=init,
            depthwise_regularizer=reg,
        )
        x = np.random.rand(2, *input_shape).astype(np.float32)
        y_before = model(x)

        save_path = os.path.join(str(tmp_path), 'bfconvunext_dw.keras')
        model.save(save_path)
        loaded = keras.models.load_model(save_path)
        y_after = loaded(x)

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(y_before),
            keras.ops.convert_to_numpy(y_after),
            atol=1e-4,
            err_msg="Outputs differ after .keras round-trip (depthwise override)"
        )


# ---------------------------------------------------------------------
# Block-activation layer-instance serialization (SC3): a multi-block
# denoiser built with a SINGLE shared LeakyReLU(0.1) instance must survive
# .keras save/load with identical outputs. This is the end-to-end guard for
# the stateless-shared-instance + layer-instance-activation path.
# ---------------------------------------------------------------------

class TestBlockActivationSerialization:
    """SC3: shared LeakyReLU(0.1) block activation .keras round-trip."""

    def test_block_activation_leaky_relu_keras_roundtrip(self, tmp_path) -> None:
        # depth>=3 per the factory guard; small but multi-block so a single
        # shared LeakyReLU instance is reused across every ConvNeXt block.
        model = create_convunext_denoiser(
            input_shape=(32, 32, 3),
            depth=3,
            initial_filters=8,
            blocks_per_level=1,
            block_activation=keras.layers.LeakyReLU(negative_slope=0.1),
        )

        rng = np.random.RandomState(1234)
        x = rng.rand(2, 32, 32, 3).astype(np.float32)
        out_before = model(x)

        save_path = os.path.join(str(tmp_path), 'denoiser_leaky.keras')
        model.save(save_path)
        reloaded = keras.models.load_model(save_path)
        out_after = reloaded(x)

        # GPU fp32 reduction noise -> atol 1e-4 (SYSTEM invariant)
        assert np.allclose(
            np.array(keras.ops.convert_to_numpy(out_before)),
            np.array(keras.ops.convert_to_numpy(out_after)),
            atol=1e-4,
        ), "Outputs differ after .keras round-trip (shared LeakyReLU(0.1))"

        # Sanity: the LeakyReLU build actually differs from the default (gelu)
        # build on the same input -> confirms the activation override took effect
        # and survived reload as a non-default activation.
        default_model = create_convunext_denoiser(
            input_shape=(32, 32, 3),
            depth=3,
            initial_filters=8,
            blocks_per_level=1,
        )
        out_default = default_model(x)
        assert not np.allclose(
            np.array(keras.ops.convert_to_numpy(out_after)),
            np.array(keras.ops.convert_to_numpy(out_default)),
            atol=1e-4,
        ), "Reloaded LeakyReLU model matches default (gelu) build; override lost"


# ---------------------------------------------------------------------
# Stem + deep-supervision activation threading (Step 9): the factory
# stem_activation / supervision_activation params must reach the stem and
# the deep-supervision heads, and a LeakyReLU(0.1) instance must survive
# .keras save/load. All save/load asserts call the model with
# training=False so StochasticDepth (stochastic in training mode) does not
# produce false mismatches.
# ---------------------------------------------------------------------

class TestStemSupervisionActivation:
    """Step 9: stem_activation / supervision_activation factory + serialization."""

    def test_stem_activation_leaky_relu_keras_roundtrip(self, tmp_path) -> None:
        # use_gabor_stem=False so the ConvUNextStem (which carries the activation)
        # is actually built at encoder level 0.
        model = create_convunext_denoiser(
            input_shape=(32, 32, 3),
            depth=3,
            initial_filters=8,
            blocks_per_level=1,
            use_gabor_stem=False,
            stem_activation=keras.layers.LeakyReLU(negative_slope=0.1),
        )

        rng = np.random.RandomState(2024)
        x = rng.rand(2, 32, 32, 3).astype(np.float32)
        out_before = model(x, training=False)

        save_path = os.path.join(str(tmp_path), 'denoiser_stem_leaky.keras')
        model.save(save_path)
        reloaded = keras.models.load_model(save_path)
        out_after = reloaded(x, training=False)

        np.testing.assert_allclose(
            np.array(keras.ops.convert_to_numpy(out_before)),
            np.array(keras.ops.convert_to_numpy(out_after)),
            atol=1e-5,
            err_msg="Outputs differ after .keras round-trip (stem LeakyReLU(0.1))",
        )

    def test_default_stem_activation_is_gelu(self) -> None:
        # Default (no stem_activation): the standard ConvUNextStem exists only when
        # use_gabor_stem=False, and its serialized activation must stay 'gelu'.
        model = create_convunext_denoiser(
            input_shape=(32, 32, 3),
            depth=3,
            initial_filters=8,
            blocks_per_level=1,
            use_gabor_stem=False,
        )
        stems = [
            l for l in model.layers
            if l.__class__.__name__ == 'ConvUNextStem'
        ]
        assert len(stems) > 0, "no ConvUNextStem found in the default build"
        for stem in stems:
            assert stem.get_config()['activation'] == 'gelu'

    def test_supervision_activation_leaky_relu_keras_roundtrip(self, tmp_path) -> None:
        model = create_convunext_denoiser(
            input_shape=(32, 32, 3),
            depth=3,
            initial_filters=8,
            blocks_per_level=1,
            enable_deep_supervision=True,
            supervision_activation=keras.layers.LeakyReLU(negative_slope=0.1),
        )

        rng = np.random.RandomState(7)
        x = rng.rand(1, 32, 32, 3).astype(np.float32)
        outs_before = model(x, training=False)
        assert isinstance(outs_before, list) and len(outs_before) > 1

        save_path = os.path.join(str(tmp_path), 'denoiser_sup_leaky.keras')
        model.save(save_path)
        reloaded = keras.models.load_model(save_path)
        outs_after = reloaded(x, training=False)

        assert len(outs_after) == len(outs_before)
        for i, (b, a) in enumerate(zip(outs_before, outs_after)):
            np.testing.assert_allclose(
                np.array(keras.ops.convert_to_numpy(b)),
                np.array(keras.ops.convert_to_numpy(a)),
                atol=1e-5,
                err_msg=f"Output {i} differs after .keras round-trip "
                        f"(supervision LeakyReLU(0.1))",
            )

    # --- Step 11: REFLECT-driven test hardening ---

    def test_full_path_shared_activation_keras_roundtrip(self, tmp_path) -> None:
        """TEST A (reviewer WARNING #2): the FULL trainer-shape model where ONE
        shared LeakyReLU(0.1) instance drives block + stem + deep-supervision
        simultaneously (use_gabor_stem=False AND enable_deep_supervision=True).

        Guards: no duplicate layer names from the shared instance, and a full
        .keras save/load identity (training=False) across ALL deep-supervision
        outputs. This mirrors build_model's real construction shape.
        """
        act = keras.layers.LeakyReLU(negative_slope=0.1)
        model = create_convunext_denoiser(
            input_shape=(32, 32, 3),
            depth=3,
            initial_filters=8,
            blocks_per_level=1,
            use_gabor_stem=False,
            enable_deep_supervision=True,
            block_activation=act,
            stem_activation=act,
            supervision_activation=act,
        )

        # No duplicate layer names (the shared instance must not collide).
        names = [l.name for l in model.layers]
        assert len(names) == len(set(names)), (
            f"duplicate layer names in shared-activation full-path build: "
            f"{[n for n in names if names.count(n) > 1]}"
        )

        # A ConvUNextStem must be present (use_gabor_stem=False) and its
        # serialized activation must deserialize to LeakyReLU(0.1).
        stems = [l for l in model.layers if l.__class__.__name__ == 'ConvUNextStem']
        assert len(stems) > 0, "no ConvUNextStem found (use_gabor_stem=False)"
        stem_act_cfg = stems[0].get_config()['activation']
        revived = keras.layers.deserialize(stem_act_cfg)
        assert isinstance(revived, keras.layers.LeakyReLU)
        assert np.isclose(float(revived.negative_slope), 0.1)

        rng = np.random.RandomState(11)
        x = rng.rand(2, 32, 32, 3).astype(np.float32)
        outs_before = model(x, training=False)
        assert isinstance(outs_before, list) and len(outs_before) > 1

        save_path = os.path.join(str(tmp_path), 'denoiser_full_shared_leaky.keras')
        model.save(save_path)
        reloaded = keras.models.load_model(save_path)
        outs_after = reloaded(x, training=False)

        assert len(outs_after) == len(outs_before)
        for i, (b, a) in enumerate(zip(outs_before, outs_after)):
            np.testing.assert_allclose(
                np.array(keras.ops.convert_to_numpy(b)),
                np.array(keras.ops.convert_to_numpy(a)),
                atol=1e-5,
                err_msg=f"Output {i} differs after .keras round-trip "
                        f"(full-path shared LeakyReLU(0.1))",
            )

    def test_leaky_vs_gelu_stem_behavioral_differential(self) -> None:
        """TEST B (reviewer WARNING #3): prove slope-0.1 is APPLIED, not merely
        serialized. Isolate the stem activation: two ConvUNextStem layers that
        differ ONLY in activation (LeakyReLU(0.1) vs 'gelu'), with identical
        weights copied across, must produce DIFFERENT outputs on a
        negative-bearing input. Plus a micro-check nailing the exact slope 0.1.
        """
        # Micro-check: the slope IS exactly 0.1 (not gelu's ~-0.158 at x=-1,
        # and not the 'leaky_relu' string default of 0.2).
        leaky_val = keras.layers.Activation(
            keras.layers.LeakyReLU(negative_slope=0.1)
        )(np.array([[-1.0]], dtype='float32'))
        leaky_val = float(np.array(keras.ops.convert_to_numpy(leaky_val))[0, 0])
        assert np.isclose(leaky_val, -0.1, atol=1e-6), (
            f"LeakyReLU(0.1) on -1.0 should be -0.1, got {leaky_val}"
        )

        # Build two stems differing only in activation.
        stem_leaky = ConvUNextStem(
            filters=8, activation=keras.layers.LeakyReLU(negative_slope=0.1)
        )
        stem_gelu = ConvUNextStem(filters=8, activation='gelu')

        rng = np.random.RandomState(99)
        # Input with a clearly-negative region so the activation branch matters.
        x = (rng.rand(2, 16, 16, 3).astype(np.float32) - 0.5) * 4.0

        # Build both on the same input shape, then copy weights so ONLY the
        # activation differs.
        _ = stem_leaky(x)
        _ = stem_gelu(x)
        stem_gelu.set_weights(stem_leaky.get_weights())

        out_leaky = np.array(keras.ops.convert_to_numpy(stem_leaky(x)))
        out_gelu = np.array(keras.ops.convert_to_numpy(stem_gelu(x)))

        # The activation actually changes behavior: outputs are NOT allclose.
        assert not np.allclose(out_leaky, out_gelu, atol=1e-4), (
            "LeakyReLU(0.1) and gelu stems produced identical outputs with "
            "identical weights -> stem activation is not being applied"
        )
        # And they differ meaningfully somewhere (not just float noise).
        assert np.max(np.abs(out_leaky - out_gelu)) > 1e-3


# ---------------------------------------------------------------------
# zero_pad_channels flag (Step 4 / SC2-SC5): parameter-free channel match.
# The flag replaces every per-level 1x1 channel-adjust conv with a weightless
# MatchChannels op (zero-pad on increase, slice+add on decrease). OFF path is
# byte-identical to HEAD. All save/load + homogeneity checks run on CPU
# (CUDA_VISIBLE_DEVICES="") per LESSONS: GPU fp32 reduction noise can exceed
# 1e-4 and mask serialization / homogeneity defects.
# ---------------------------------------------------------------------

class TestZeroPadChannels:
    """Step 4: model-level tests for the zero_pad_channels factory flag."""

    # Small, fast, homogeneity-friendly config: v1 (no GRN), linear final
    # activation (homogeneity), filter_multiplier=2 so channel-adjust sites
    # actually exist (and get removed when ON).
    def _cfg(self, **overrides) -> Dict[str, Any]:
        cfg = dict(
            input_shape=(32, 32, 1),
            depth=3,
            initial_filters=16,
            blocks_per_level=1,
            convnext_version='v1',
            filter_multiplier=2,
            final_activation='linear',
            drop_path_rate=0.0,
        )
        cfg.update(overrides)
        return cfg

    def _build(self, **overrides) -> keras.Model:
        return create_convunext_denoiser(**self._cfg(**overrides))

    def test_zero_pad_default_unchanged(self) -> None:
        # SC3: explicit OFF and the omitted-kwarg default must be byte-identical
        # (same layer-name SET, same outputs to atol=0).
        model_explicit = self._build(zero_pad_channels=False)
        model_default = self._build()

        names_explicit = {l.name for l in model_explicit.layers}
        names_default = {l.name for l in model_default.layers}
        assert names_explicit == names_default, (
            "explicit zero_pad_channels=False diverged from the omitted-kwarg "
            f"default: symmetric diff {names_explicit ^ names_default}"
        )

        rng = np.random.RandomState(0)
        x = rng.rand(2, 32, 32, 1).astype(np.float32)
        # Same seed/init is not guaranteed across two builds, so compare the
        # GRAPH topology (names) for byte-identity; for output identity, copy
        # weights so only the wiring is under test.
        model_default.set_weights(model_explicit.get_weights())
        y_explicit = np.array(keras.ops.convert_to_numpy(model_explicit(x, training=False)))
        y_default = np.array(keras.ops.convert_to_numpy(model_default(x, training=False)))
        np.testing.assert_allclose(
            y_explicit, y_default, atol=0.0,
            err_msg="OFF explicit vs omitted-kwarg outputs differ (atol=0)",
        )

    def test_zero_pad_forward_shape(self) -> None:
        # SC2: base-like AND tiny-like ON configs build, forward-pass to the same
        # spatial+channel shape as the input, and are all-finite.
        for name, cfg in (
            ("base-like", dict(initial_filters=16, depth=3)),
            ("tiny-like", dict(initial_filters=8, depth=3)),
        ):
            model = self._build(zero_pad_channels=True, **cfg)
            rng = np.random.RandomState(1)
            x = rng.rand(2, 32, 32, 1).astype(np.float32)
            y = np.array(keras.ops.convert_to_numpy(model(x, training=False)))
            assert y.shape == x.shape, f"{name}: output {y.shape} != input {x.shape}"
            assert np.all(np.isfinite(y)), f"{name}: non-finite ON forward output"

    def test_zero_pad_param_count_drops(self) -> None:
        # SC2: removing the channel-adjust convs strictly reduces param count
        # (filter_multiplier=2 guarantees the convs exist in the OFF model).
        off = self._build(zero_pad_channels=False)
        on = self._build(zero_pad_channels=True)
        assert on.count_params() < off.count_params(), (
            f"ON params ({on.count_params()}) not < OFF params "
            f"({off.count_params()}); channel-adjust convs were not removed"
        )

    def test_zero_pad_bias_free(self) -> None:
        # SC5 (bias-free arm): verify_bias_free is a logging helper (returns None,
        # never raises); confirm it runs, then INDEPENDENTLY assert the ON model
        # carries no bias and no LayerNormalization center. v1 has no GRN beta, so
        # a correct ON model has ZERO bias-like offenders.
        from train.bfunet.train_convunext_denoiser import verify_bias_free

        model = self._build(zero_pad_channels=True)
        # Smoke: the helper executes on the ON model without error.
        assert verify_bias_free(model) is None

        offenders = []
        for layer in model._flatten_layers():
            if getattr(layer, "use_bias", False):
                offenders.append(layer.name)
            if isinstance(layer, keras.layers.LayerNormalization) and getattr(
                layer, "center", False
            ):
                offenders.append(f"{layer.name} (LN center)")
        assert not offenders, f"ON model is not bias-free; offenders: {offenders}"

    def test_zero_pad_keras_round_trip(self, tmp_path) -> None:
        # SC4: ON model .keras save/load identity on CPU (atol=1e-4). MatchChannels
        # is registered, but pass it in custom_objects defensively (mirrors the
        # LESSONS guidance for the model package + custom layers).
        from dl_techniques.layers.match_channels import MatchChannels

        model = self._build(zero_pad_channels=True)
        rng = np.random.RandomState(2)
        x = rng.rand(2, 32, 32, 1).astype(np.float32)
        y_before = np.array(keras.ops.convert_to_numpy(model(x, training=False)))

        save_path = os.path.join(str(tmp_path), 'bfconvunext_zeropad.keras')
        model.save(save_path)
        loaded = keras.models.load_model(
            save_path, custom_objects={'MatchChannels': MatchChannels}
        )
        y_after = np.array(keras.ops.convert_to_numpy(loaded(x, training=False)))

        np.testing.assert_allclose(
            y_before, y_after, atol=1e-4,
            err_msg="ON model outputs differ after .keras round-trip",
        )

    def test_zero_pad_homogeneity_differential(self) -> None:
        # SC5 (differential homogeneity arm): with the trainer's homogeneous
        # config (LeakyReLU(0.1) on block + stem + linear final), the ON
        # parameter-free pad/slice must NOT break the scale homogeneity the OFF
        # path has. Measure err = max|f(alpha*x) - alpha*f(x)| for both; assert
        # ON err <= OFF err * 1.5 + 1e-4. Both expected ~1e-4 on CPU.
        alpha = 3.0
        rng = np.random.RandomState(3)
        # negative-bearing input so LeakyReLU's negative branch is exercised
        x = ((rng.rand(2, 32, 32, 1).astype(np.float32) - 0.5) * 2.0)

        def _hom_err(model: keras.Model) -> float:
            fx = np.array(keras.ops.convert_to_numpy(model(x, training=False)))
            fax = np.array(keras.ops.convert_to_numpy(model(alpha * x, training=False)))
            return float(np.max(np.abs(fax - alpha * fx)))

        # ONE shared activation instance drives block + stem (mirrors build_model).
        def _homog_cfg(zero_pad: bool) -> keras.Model:
            act = keras.layers.LeakyReLU(negative_slope=0.1)
            return self._build(
                zero_pad_channels=zero_pad,
                use_gabor_stem=False,
                block_activation=act,
                stem_activation=act,
            )

        off_model = _homog_cfg(False)
        on_model = _homog_cfg(True)

        off_err = _hom_err(off_model)
        on_err = _hom_err(on_model)

        assert on_err <= off_err * 1.5 + 1e-4, (
            f"ON homogeneity err {on_err:.3e} exceeds OFF {off_err:.3e} * 1.5 "
            f"+ 1e-4; parameter-free pad/slice broke scale homogeneity"
        )


# ---------------------------------------------------------------------
# create_convunext_denoiser(..., extra_zero_output_channels=True)
# ---------------------------------------------------------------------
# Design B (D-001/D-002): at decoder level 0, zero-pad x to
# initial_filters + output_channels BEFORE the level-0 ConvNeXt blocks
# (widened blocks), then drop the learned final_output 1x1 Conv2D in favor
# of a parameter-free tail-slice keeping the last output_channels channels
# (MatchChannels(slice_side='tail')). OFF path is byte-identical to HEAD.
# All save/load + homogeneity checks run on CPU (CUDA_VISIBLE_DEVICES="")
# per LESSONS: GPU fp32 reduction noise can mask serialization / homogeneity
# defects at 1e-4.
# ---------------------------------------------------------------------

class TestExtraZeroOutputChannels:
    """Step 3: model-level tests for the extra_zero_output_channels factory flag."""

    # Same small, fast, homogeneity-friendly config as TestZeroPadChannels.
    def _cfg(self, **overrides) -> Dict[str, Any]:
        cfg = dict(
            input_shape=(32, 32, 1),
            depth=3,
            initial_filters=16,
            blocks_per_level=1,
            convnext_version='v1',
            filter_multiplier=2,
            final_activation='linear',
            drop_path_rate=0.0,
        )
        cfg.update(overrides)
        return cfg

    def _build(self, **overrides) -> keras.Model:
        return create_convunext_denoiser(**self._cfg(**overrides))

    def test_extra_zero_default_unchanged(self) -> None:
        # SC4: explicit OFF and the omitted-kwarg default must be byte-identical
        # (same layer-name SET, same outputs to atol=0).
        model_explicit = self._build(extra_zero_output_channels=False)
        model_default = self._build()

        names_explicit = {l.name for l in model_explicit.layers}
        names_default = {l.name for l in model_default.layers}
        assert names_explicit == names_default, (
            "explicit extra_zero_output_channels=False diverged from the "
            f"omitted-kwarg default: symmetric diff {names_explicit ^ names_default}"
        )

        rng = np.random.RandomState(0)
        x = rng.rand(2, 32, 32, 1).astype(np.float32)
        # Same seed/init is not guaranteed across two builds, so compare the
        # GRAPH topology (names) for byte-identity; for output identity, copy
        # weights so only the wiring is under test.
        model_default.set_weights(model_explicit.get_weights())
        y_explicit = np.array(keras.ops.convert_to_numpy(model_explicit(x, training=False)))
        y_default = np.array(keras.ops.convert_to_numpy(model_default(x, training=False)))
        np.testing.assert_allclose(
            y_explicit, y_default, atol=0.0,
            err_msg="OFF explicit vs omitted-kwarg outputs differ (atol=0)",
        )

    def test_extra_zero_forward_shape(self) -> None:
        # SC5: base-like AND tiny-like ON configs build, forward-pass to the same
        # spatial+channel shape as the input, and are all-finite.
        for name, cfg in (
            ("base-like", dict(initial_filters=16, depth=3)),
            ("tiny-like", dict(initial_filters=8, depth=3)),
        ):
            model = self._build(extra_zero_output_channels=True, **cfg)
            rng = np.random.RandomState(1)
            x = rng.rand(2, 32, 32, 1).astype(np.float32)
            y = np.array(keras.ops.convert_to_numpy(model(x, training=False)))
            assert y.shape == x.shape, f"{name}: output {y.shape} != input {x.shape}"
            assert np.all(np.isfinite(y)), f"{name}: non-finite ON forward output"

    def test_extra_zero_final_output_absent(self) -> None:
        # SC6: the learned 1x1 'final_output' Conv2D is present OFF and DROPPED ON,
        # replaced by the parameter-free 'final_output_tail_slice' MatchChannels;
        # the level-0 zero-pad layer 'extra_zero_output_pad' is added ON.
        # NOTE: we do NOT assert a net param-count drop here — the widened level-0
        # ConvNeXt blocks ADD params; the invariant under test is the SPECIFIC
        # final_output-Conv2D removal, not the net delta.
        off = self._build(extra_zero_output_channels=False)
        on = self._build(extra_zero_output_channels=True)

        off_names = {l.name for l in off.layers}
        on_names = {l.name for l in on.layers}

        assert 'final_output' in off_names, "OFF model is missing final_output Conv2D"
        assert 'final_output' not in on_names, (
            "ON model still carries the learned final_output Conv2D"
        )
        assert 'final_output_tail_slice' in on_names, (
            "ON model is missing the parameter-free final_output_tail_slice"
        )
        assert 'extra_zero_output_pad' in on_names, (
            "ON model is missing the level-0 extra_zero_output_pad"
        )

    def test_extra_zero_bias_free(self) -> None:
        # SC7 (bias-free arm): verify_bias_free is a logging helper (returns None,
        # never raises); confirm it runs, then INDEPENDENTLY assert the ON model
        # carries no bias and no LayerNormalization center.
        from train.bfunet.train_convunext_denoiser import verify_bias_free

        model = self._build(extra_zero_output_channels=True)
        # Smoke: the helper executes on the ON model without error.
        assert verify_bias_free(model) is None

        offenders = []
        for layer in model._flatten_layers():
            if getattr(layer, "use_bias", False):
                offenders.append(layer.name)
            if isinstance(layer, keras.layers.LayerNormalization) and getattr(
                layer, "center", False
            ):
                offenders.append(f"{layer.name} (LN center)")
        assert not offenders, f"ON model is not bias-free; offenders: {offenders}"

    def test_extra_zero_keras_round_trip(self, tmp_path) -> None:
        # SC8: ON model .keras save/load identity on CPU (atol=1e-4). MatchChannels
        # is registered, but pass it in custom_objects defensively.
        from dl_techniques.layers.match_channels import MatchChannels

        model = self._build(extra_zero_output_channels=True)
        rng = np.random.RandomState(2)
        x = rng.rand(2, 32, 32, 1).astype(np.float32)
        y_before = np.array(keras.ops.convert_to_numpy(model(x, training=False)))

        save_path = os.path.join(str(tmp_path), 'bfconvunext_extra_zero.keras')
        model.save(save_path)
        loaded = keras.models.load_model(
            save_path, custom_objects={'MatchChannels': MatchChannels}
        )
        y_after = np.array(keras.ops.convert_to_numpy(loaded(x, training=False)))

        np.testing.assert_allclose(
            y_before, y_after, atol=1e-4,
            err_msg="ON model outputs differ after .keras round-trip",
        )

    def test_extra_zero_homogeneity_differential(self) -> None:
        # SC7 (differential homogeneity arm): with the trainer's homogeneous
        # config (LeakyReLU(0.1) on block + stem + linear final), the ON
        # parameter-free pad/tail-slice must NOT break the scale homogeneity the
        # OFF path has. Measure err = max|f(alpha*x) - alpha*f(x)| for both;
        # assert ON err <= OFF err * 1.5 + 1e-4. Both expected ~1e-4 on CPU.
        alpha = 3.0
        rng = np.random.RandomState(3)
        # negative-bearing input so LeakyReLU's negative branch is exercised
        x = ((rng.rand(2, 32, 32, 1).astype(np.float32) - 0.5) * 2.0)

        def _hom_err(model: keras.Model) -> float:
            fx = np.array(keras.ops.convert_to_numpy(model(x, training=False)))
            fax = np.array(keras.ops.convert_to_numpy(model(alpha * x, training=False)))
            return float(np.max(np.abs(fax - alpha * fx)))

        # ONE shared activation instance drives block + stem (mirrors build_model).
        def _homog_cfg(extra_zero: bool) -> keras.Model:
            act = keras.layers.LeakyReLU(negative_slope=0.1)
            return self._build(
                extra_zero_output_channels=extra_zero,
                use_gabor_stem=False,
                block_activation=act,
                stem_activation=act,
            )

        off_model = _homog_cfg(False)
        on_model = _homog_cfg(True)

        off_err = _hom_err(off_model)
        on_err = _hom_err(on_model)

        assert on_err <= off_err * 1.5 + 1e-4, (
            f"ON homogeneity err {on_err:.3e} exceeds OFF {off_err:.3e} * 1.5 "
            f"+ 1e-4; parameter-free pad/tail-slice broke scale homogeneity"
        )

    def test_extra_zero_compose_with_zero_pad(self) -> None:
        # SC9 (compose arm): extra_zero_output_channels + zero_pad_channels both ON
        # must build and forward to the input shape, all-finite.
        model = self._build(
            extra_zero_output_channels=True, zero_pad_channels=True
        )
        rng = np.random.RandomState(4)
        x = rng.rand(2, 32, 32, 1).astype(np.float32)
        y = np.array(keras.ops.convert_to_numpy(model(x, training=False)))
        assert y.shape == (2, 32, 32, 1), f"compose output {y.shape} != (2,32,32,1)"
        assert np.all(np.isfinite(y)), "non-finite compose (zero_pad) forward output"

    def test_extra_zero_compose_with_deep_supervision(self) -> None:
        # SC9 (compose arm): extra_zero_output_channels + enable_deep_supervision
        # both ON builds a valid multi-output model; the PRIMARY output (index 0)
        # has the input shape and is finite.
        model = self._build(
            extra_zero_output_channels=True, enable_deep_supervision=True
        )
        rng = np.random.RandomState(5)
        x = rng.rand(2, 32, 32, 1).astype(np.float32)
        outputs = model(x, training=False)
        assert isinstance(outputs, (list, tuple)), (
            f"deep-supervision model is not multi-output: {type(outputs)}"
        )
        assert len(outputs) >= 2, f"expected >=2 outputs, got {len(outputs)}"
        primary = np.array(keras.ops.convert_to_numpy(outputs[0]))
        assert primary.shape == (2, 32, 32, 1), (
            f"primary output {primary.shape} != (2,32,32,1)"
        )
        assert np.all(np.isfinite(primary)), "non-finite primary deep-supervision output"


# ---------------------------------------------------------------------
# create_convunext_variant (wrapper)
# ---------------------------------------------------------------------

class TestCreateConvUNextVariant:
    """Test suite for the variant wrapper."""

    def test_tiny_variant(self) -> None:
        model = create_convunext_variant(
            'tiny', (32, 32, 1), enable_deep_supervision=False
        )
        x = np.random.rand(1, 32, 32, 1).astype(np.float32)
        y = model(x)
        assert y.shape == (1, 32, 32, 1)

    def test_unknown_variant_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown variant"):
            create_convunext_variant('nonexistent', (32, 32, 1))

    def test_all_variants_registered(self) -> None:
        assert set(CONVUNEXT_CONFIGS) >= {'tiny', 'small', 'base', 'large', 'xlarge'}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
