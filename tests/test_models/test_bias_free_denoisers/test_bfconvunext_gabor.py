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
        y_before = model(x)

        save_path = os.path.join(str(tmp_path), 'bfconvunext_gabor.keras')
        model.save(save_path)
        loaded = keras.models.load_model(save_path)
        y_after = loaded(x)

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
