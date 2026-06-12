"""Tiny smoke / integration tests for the Ideogram4 inference pipeline.

Uses the TINY preset with a deliberately small image so the grid is 2x2 (4
image tokens), T=3 text tokens, num_steps=2, B=1 -- the whole denoise+decode
runs in seconds. Covers:

- ``_build_inputs`` correctness: indicator counts, position_id image offset,
  packed shapes.
- ``apply_cfg_blend`` math (gw=1 -> conditional only; gw=0 -> unconditional;
  linear blend in between).
- A full ``__call__`` denoise+decode producing a finite image of the expected
  shape and value range.
- Determinism: same seed -> identical image; different seed -> different.
- Both a constant ``guidance_scale`` and an explicit ``guidance_schedule``.

Conditioning is the precomputed ``llm_features`` INPUT (decision D1).
"""

import numpy as np
import keras
import pytest

from dl_techniques.models.ideogram4.config import get_ideogram4_config
from dl_techniques.models.ideogram4.constants import (
    IMAGE_POSITION_OFFSET,
    LLM_TOKEN_INDICATOR,
    OUTPUT_IMAGE_INDICATOR,
)
from dl_techniques.models.ideogram4.pipeline import (
    Ideogram4Pipeline,
    apply_cfg_blend,
)

# ---------------------------------------------------------------------
# Tiny geometry. pixels_per_token_edge = patch_size(2) * vae_factor(2) = 4.
# height = width = 4 * 2 = 8 -> grid 2x2 -> 4 image tokens.
# ---------------------------------------------------------------------
BATCH = 1
NUM_TEXT = 3
HEIGHT = 8
WIDTH = 8
NUM_STEPS = 2
GRID = 2
NUM_IMAGE = GRID * GRID  # 4


@pytest.fixture(scope="module")
def tiny_config():
    return get_ideogram4_config("tiny")


@pytest.fixture(scope="module")
def pipeline():
    return Ideogram4Pipeline.from_config("tiny", seed=0)


@pytest.fixture
def llm_features(tiny_config):
    config, _ = tiny_config
    rng = np.random.default_rng(123)
    return rng.standard_normal(
        (BATCH, NUM_TEXT, config.llm_features_dim)
    ).astype("float32")


# ---------------------------------------------------------------------
# apply_cfg_blend (isolated math)
# ---------------------------------------------------------------------
class TestCFGBlend:
    def test_gw_one_is_conditional_only(self):
        pos = keras.ops.convert_to_tensor(
            np.random.randn(2, 4, 8).astype("float32")
        )
        neg = keras.ops.convert_to_tensor(
            np.random.randn(2, 4, 8).astype("float32")
        )
        out = apply_cfg_blend(pos, neg, 1.0)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(out),
            keras.ops.convert_to_numpy(pos),
            atol=1e-6,
        )

    def test_gw_zero_is_unconditional_only(self):
        pos = keras.ops.convert_to_tensor(
            np.random.randn(2, 4, 8).astype("float32")
        )
        neg = keras.ops.convert_to_tensor(
            np.random.randn(2, 4, 8).astype("float32")
        )
        out = apply_cfg_blend(pos, neg, 0.0)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(out),
            keras.ops.convert_to_numpy(neg),
            atol=1e-6,
        )

    def test_linear_blend(self):
        pos = np.random.randn(2, 4, 8).astype("float32")
        neg = np.random.randn(2, 4, 8).astype("float32")
        gw = 7.0
        out = apply_cfg_blend(
            keras.ops.convert_to_tensor(pos),
            keras.ops.convert_to_tensor(neg),
            gw,
        )
        expected = gw * pos + (1.0 - gw) * neg
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(out), expected, atol=1e-5
        )


# ---------------------------------------------------------------------
# _build_inputs correctness
# ---------------------------------------------------------------------
class TestBuildInputs:
    def test_geometry_and_shapes(self, pipeline):
        assert pipeline.vae_upsample_factor == 2
        assert pipeline.pixels_per_token_edge == 4

        (
            position_ids,
            segment_ids,
            indicator,
            num_image,
            grid_h,
            grid_w,
        ) = pipeline._build_inputs(BATCH, NUM_TEXT, HEIGHT, WIDTH)

        total_len = NUM_TEXT + NUM_IMAGE
        assert (grid_h, grid_w) == (GRID, GRID)
        assert num_image == NUM_IMAGE
        assert position_ids.shape == (BATCH, total_len, 3)
        assert segment_ids.shape == (BATCH, total_len)
        assert indicator.shape == (BATCH, total_len)

    def test_indicator_split(self, pipeline):
        _, _, indicator, _, _, _ = pipeline._build_inputs(
            BATCH, NUM_TEXT, HEIGHT, WIDTH
        )
        text_part = indicator[0, :NUM_TEXT]
        image_part = indicator[0, NUM_TEXT:]
        assert np.all(text_part == LLM_TOKEN_INDICATOR)
        assert np.all(image_part == OUTPUT_IMAGE_INDICATOR)
        assert (text_part == LLM_TOKEN_INDICATOR).sum() == NUM_TEXT
        assert (image_part == OUTPUT_IMAGE_INDICATOR).sum() == NUM_IMAGE

    def test_position_ids_image_offset(self, pipeline):
        position_ids, _, _, _, _, _ = pipeline._build_inputs(
            BATCH, NUM_TEXT, HEIGHT, WIDTH
        )
        text_pos = position_ids[0, :NUM_TEXT]
        image_pos = position_ids[0, NUM_TEXT:]

        # Text: arange(T) replicated across 3 axes, no offset.
        for axis in range(3):
            np.testing.assert_array_equal(
                text_pos[:, axis], np.arange(NUM_TEXT)
            )

        # Image: every entry >= IMAGE_POSITION_OFFSET; t-axis exactly offset.
        assert np.all(image_pos >= IMAGE_POSITION_OFFSET)
        np.testing.assert_array_equal(
            image_pos[:, 0], np.full(NUM_IMAGE, IMAGE_POSITION_OFFSET)
        )
        # h/w grid coordinates (row-major over a 2x2 grid).
        hh = np.array([0, 0, 1, 1]) + IMAGE_POSITION_OFFSET
        ww = np.array([0, 1, 0, 1]) + IMAGE_POSITION_OFFSET
        np.testing.assert_array_equal(image_pos[:, 1], hh)
        np.testing.assert_array_equal(image_pos[:, 2], ww)

    def test_segment_ids_single_segment(self, pipeline):
        _, segment_ids, _, _, _, _ = pipeline._build_inputs(
            BATCH, NUM_TEXT, HEIGHT, WIDTH
        )
        assert np.all(segment_ids == 1)


# ---------------------------------------------------------------------
# Full denoise + decode smoke
# ---------------------------------------------------------------------
class TestPipelineCall:
    def _shape(self):
        config, ae = get_ideogram4_config("tiny")
        return (BATCH, HEIGHT, WIDTH, ae.out_ch)

    def test_finite_image_constant_guidance(self, pipeline, llm_features):
        img = pipeline(
            llm_features=llm_features,
            height=HEIGHT,
            width=WIDTH,
            num_steps=NUM_STEPS,
            guidance_scale=7.0,
            seed=0,
        )
        arr = keras.ops.convert_to_numpy(img)
        assert arr.shape == self._shape()
        assert np.all(np.isfinite(arr)), "image has NaN/Inf"
        assert arr.min() >= 0.0 - 1e-5 and arr.max() <= 1.0 + 1e-5

    def test_guidance_schedule(self, pipeline, llm_features):
        img = pipeline(
            llm_features=llm_features,
            height=HEIGHT,
            width=WIDTH,
            num_steps=NUM_STEPS,
            guidance_schedule=(3.0, 7.0),  # len == num_steps
            seed=0,
        )
        arr = keras.ops.convert_to_numpy(img)
        assert arr.shape == self._shape()
        assert np.all(np.isfinite(arr))

    def test_guidance_schedule_length_mismatch_raises(
        self, pipeline, llm_features
    ):
        with pytest.raises(ValueError):
            pipeline(
                llm_features=llm_features,
                height=HEIGHT,
                width=WIDTH,
                num_steps=NUM_STEPS,
                guidance_schedule=(3.0, 7.0, 5.0),  # wrong length
                seed=0,
            )

    def test_bad_resolution_raises(self, pipeline, llm_features):
        with pytest.raises(ValueError):
            pipeline(
                llm_features=llm_features,
                height=HEIGHT + 1,  # not divisible by 4
                width=WIDTH,
                num_steps=NUM_STEPS,
                seed=0,
            )


# ---------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------
class TestDeterminism:
    def test_same_seed_identical(self, pipeline, llm_features):
        a = keras.ops.convert_to_numpy(
            pipeline(
                llm_features=llm_features,
                height=HEIGHT,
                width=WIDTH,
                num_steps=NUM_STEPS,
                guidance_scale=7.0,
                seed=42,
            )
        )
        b = keras.ops.convert_to_numpy(
            pipeline(
                llm_features=llm_features,
                height=HEIGHT,
                width=WIDTH,
                num_steps=NUM_STEPS,
                guidance_scale=7.0,
                seed=42,
            )
        )
        np.testing.assert_allclose(a, b, atol=1e-6)

    def test_different_seed_differs(self, pipeline, llm_features):
        a = keras.ops.convert_to_numpy(
            pipeline(
                llm_features=llm_features,
                height=HEIGHT,
                width=WIDTH,
                num_steps=NUM_STEPS,
                guidance_scale=7.0,
                seed=1,
            )
        )
        b = keras.ops.convert_to_numpy(
            pipeline(
                llm_features=llm_features,
                height=HEIGHT,
                width=WIDTH,
                num_steps=NUM_STEPS,
                guidance_scale=7.0,
                seed=2,
            )
        )
        assert not np.allclose(a, b, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
