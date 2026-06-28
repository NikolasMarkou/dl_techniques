"""
Test suite for the optional frozen Gabor depthwise stem of the bias-free ConvUNext
denoiser (models/bias_free_denoisers/bfconvunext.py, plan_2026-06-19_ed071c02/D-001).

Covers: build with use_gabor_stem=True (full-resolution output preserved), the stem
being non-learnable (trainable=False, zero trainable parameters), the mandatory
bias-free 1x1 projection, default use_gabor_stem=False producing the unchanged
architecture (no Gabor layer), and full .keras save -> load -> identical-output
round-trip with the Gabor stem present.
"""

import os
import keras
import numpy as np
from typing import Tuple

from dl_techniques.models.bias_free_denoisers.bfconvunext import (
    create_convunext_denoiser,
    create_convunext_variant,
)


# ---------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------

INPUT_SHAPE: Tuple[int, int, int] = (64, 64, 3)


def _build_gabor(input_shape=INPUT_SHAPE, **overrides):
    cfg = dict(
        input_shape=input_shape,
        depth=3,
        initial_filters=16,
        blocks_per_level=1,
        convnext_version='v1',
        use_gabor_stem=True,
        gabor_filters=8,
        gabor_kernel_size=7,
    )
    cfg.update(overrides)
    return create_convunext_denoiser(**cfg)


# ---------------------------------------------------------------------
# Gabor stem
# ---------------------------------------------------------------------

class TestGaborStem:
    """Frozen Gabor depthwise stem on create_convunext_denoiser."""

    def test_build_and_full_res_output(self) -> None:
        model = _build_gabor()
        x = np.random.randn(2, *INPUT_SHAPE).astype(np.float32)
        y = model(x)
        # Denoiser output must preserve full spatial resolution and channel count.
        assert tuple(y.shape) == (2, *INPUT_SHAPE)

    def test_gabor_layer_present_and_frozen(self) -> None:
        model = _build_gabor()
        gabor = model.get_layer('gabor_stem')
        # Non-learnable: frozen flag set and zero trainable parameters.
        assert gabor.trainable is False
        assert len(gabor.trainable_weights) == 0
        n_trainable = int(sum(np.prod(w.shape) for w in gabor.trainable_weights))
        assert n_trainable == 0

    def test_projection_is_bias_free(self) -> None:
        model = _build_gabor()
        proj = model.get_layer('gabor_stem_projection')
        # Mandatory 1x1 projection (in_ch * gabor_filters -> initial_filters), bias-free.
        assert proj.use_bias is False
        assert proj.filters == 16

    def test_gabor_output_channels(self) -> None:
        # Depthwise output channels = in_channels * gabor_filters = 3 * 8 = 24.
        model = _build_gabor()
        gabor = model.get_layer('gabor_stem')
        out_shape = gabor.compute_output_shape((None, 64, 64, 3))
        assert out_shape[-1] == 3 * 8

    def test_default_has_no_gabor(self) -> None:
        model = create_convunext_denoiser(
            input_shape=INPUT_SHAPE,
            depth=3,
            initial_filters=16,
            blocks_per_level=1,
            convnext_version='v1',
        )
        names = [l.name for l in model.layers]
        assert 'gabor_stem' not in names
        assert 'gabor_stem_projection' not in names

    def test_variant_forwards_gabor_kwargs(self) -> None:
        model = create_convunext_variant(
            'tiny', INPUT_SHAPE, enable_deep_supervision=False,
            convnext_version='v1', use_gabor_stem=True, gabor_filters=8,
        )
        assert 'gabor_stem' in [l.name for l in model.layers]

    def test_keras_round_trip(self, tmp_path) -> None:
        model = _build_gabor()
        x = np.random.rand(2, *INPUT_SHAPE).astype(np.float32)
        # training=False: StochasticDepth (drop-path) is identity at inference, so
        # the forward is deterministic and round-trip equality is meaningful.
        y_before = model(x, training=False)

        save_path = os.path.join(str(tmp_path), 'bfconvunext_gabor.keras')
        model.save(save_path)
        loaded = keras.models.load_model(save_path)
        y_after = loaded(x, training=False)

        # Gabor stem must survive serialization and remain frozen after reload.
        gabor = loaded.get_layer('gabor_stem')
        assert gabor.trainable is False
        assert len(gabor.trainable_weights) == 0

        # GPU fp32 reduction noise -> atol 1e-4 (SYSTEM invariant).
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(y_before),
            keras.ops.convert_to_numpy(y_after),
            atol=1e-4,
            err_msg="Outputs differ after .keras round-trip with Gabor stem",
        )


class TestGaborStemNoProjection:
    """No-projection Gabor stem (gabor_stem_projection=False): the depthwise bank feeds
    the encoder directly, valid only when input_channels*gabor_filters == initial_filters."""

    def test_no_projection_layer_and_full_res_output(self) -> None:
        # 3 channels * 8 filters == 24 -> initial_filters must be 24.
        model = _build_gabor(gabor_stem_projection=False, initial_filters=24)
        names = [l.name for l in model.layers]
        assert 'gabor_stem' in names, "Gabor stem layer should still be present"
        assert 'gabor_stem_projection' not in names, "projection must be dropped"
        x = np.random.uniform(-0.5, 0.5, size=(2, *INPUT_SHAPE)).astype("float32")
        y = model(x, training=False)
        assert tuple(y.shape) == (2, *INPUT_SHAPE)

    def test_gabor_feeds_initial_filters_channels(self) -> None:
        model = _build_gabor(gabor_stem_projection=False, initial_filters=24)
        gabor = model.get_layer('gabor_stem')
        # Depthwise bank output: in_channels(3) * gabor_filters(8) == initial_filters(24).
        assert gabor.output.shape[-1] == 24

    def test_no_projection_is_bias_free(self) -> None:
        # Removing the projection must not introduce any bias / centering anywhere.
        model = _build_gabor(gabor_stem_projection=False, initial_filters=24)
        offenders = [
            l.name for l in model._flatten_layers()
            if getattr(l, "use_bias", False)
            or (isinstance(l, keras.layers.LayerNormalization) and getattr(l, "center", False))
        ]
        assert offenders == [], f"bias/centering survived: {offenders}"

    def test_channel_mismatch_raises(self) -> None:
        # 3 * 8 = 24 != 16 -> must fail loudly rather than silently pad/slice.
        import pytest
        with pytest.raises(ValueError, match="initial_filters"):
            _build_gabor(gabor_stem_projection=False, initial_filters=16)

    def test_default_keeps_projection(self) -> None:
        # gabor_stem_projection defaults True -> projection present, byte-identical path.
        model = _build_gabor()  # initial_filters=16, projection reduces 24 -> 16
        names = [l.name for l in model.layers]
        assert 'gabor_stem_projection' in names


class TestResidualStochasticDepth:
    """Regression guard: ConvNeXt blocks MUST be wired as residual branches with
    stochastic depth (the factory bug where blocks were chained sequentially with
    no residual / no drop-path must not recur)."""

    def test_residual_and_stochastic_depth_present(self):
        from dl_techniques.layers.stochastic_depth import StochasticDepth
        model = create_convunext_denoiser(
            input_shape=(64, 64, 3), depth=3, initial_filters=16,
            blocks_per_level=2, convnext_version="v1", drop_path_rate=0.2,
        )
        n_add = sum(1 for l in model.layers if isinstance(l, keras.layers.Add))
        n_sd = sum(1 for l in model.layers if isinstance(l, StochasticDepth))
        # depth=3 -> 3 enc + 1 bottleneck + 3 dec block-groups, 2 blocks each = 14 blocks.
        assert n_add == 14, f"expected 14 residual adds, got {n_add}"
        assert n_sd >= 1, "no StochasticDepth (drop-path) layers found"

    def test_no_residual_when_blocks_chained_would_change_shape_is_safe(self):
        # Residual add requires matching channels; the factory channel-adjusts before
        # blocks, so output must still equal the full-res input.
        model = create_convunext_denoiser(
            input_shape=(64, 64, 3), depth=3, initial_filters=16,
            blocks_per_level=1, convnext_version="v2", drop_path_rate=0.1,
        )
        y = model(np.zeros((2, 64, 64, 3), "float32"), training=False)
        assert tuple(y.shape) == (2, 64, 64, 3)

    def test_drop_path_zero_has_no_stochastic_depth(self):
        from dl_techniques.layers.stochastic_depth import StochasticDepth
        model = create_convunext_denoiser(
            input_shape=(64, 64, 3), depth=3, initial_filters=16,
            blocks_per_level=1, convnext_version="v1", drop_path_rate=0.0,
        )
        n_sd = sum(1 for l in model.layers if isinstance(l, StochasticDepth))
        n_add = sum(1 for l in model.layers if isinstance(l, keras.layers.Add))
        assert n_sd == 0, "drop_path_rate=0 should add no StochasticDepth layers"
        assert n_add >= 1, "residual connections must exist even at drop_path_rate=0"
