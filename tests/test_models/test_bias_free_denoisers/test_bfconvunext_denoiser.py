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
