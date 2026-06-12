"""Tests for Ideogram4 config, constants, and latent normalization.

Covers (per plan step 5):
- All ``__post_init__`` / build-time invariants RAISE ValueError on bad dims.
- ``tiny`` and ``full`` presets BUILD and satisfy all invariants.
- ``to_dict`` / ``from_dict`` round-trip for both dataclasses (tuple<->list).
- ``get_latent_norm`` shapes + value spot-checks.
- ``constants`` exact values + QWEN3_VL_ACTIVATION_LAYERS length.
"""

import dataclasses

import keras
import numpy as np
import pytest

from dl_techniques.models.ideogram4 import constants
from dl_techniques.models.ideogram4.latent_norm import (
    LATENT_SCALE,
    LATENT_SHIFT,
    get_latent_norm,
)
from dl_techniques.models.ideogram4.config import (
    AutoEncoderParams,
    Ideogram4Config,
    PRESETS,
    get_ideogram4_config,
)


# A baseline of valid tiny kwargs reused for crafting single-field violations.
_TINY = dict(PRESETS["tiny"]["config"])


class TestConstants:
    def test_exact_values(self):
        assert constants.SEQUENCE_PADDING_INDICATOR == -1
        assert constants.OUTPUT_IMAGE_INDICATOR == 2
        assert constants.LLM_TOKEN_INDICATOR == 3
        assert constants.IMAGE_POSITION_OFFSET == 65536
        assert constants.QWEN3_VL_ACTIVATION_LAYERS == (
            0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35,
        )

    def test_activation_layers_length_13(self):
        assert len(constants.QWEN3_VL_ACTIVATION_LAYERS) == 13


class TestLatentNorm:
    def test_shapes(self):
        shift, scale = get_latent_norm()
        assert tuple(keras.ops.shape(shift)) == (128,)
        assert tuple(keras.ops.shape(scale)) == (128,)

    def test_tuple_lengths(self):
        assert len(LATENT_SHIFT) == 128
        assert len(LATENT_SCALE) == 128

    def test_first_values_match_source(self):
        shift, scale = get_latent_norm()
        shift = keras.ops.convert_to_numpy(shift)
        scale = keras.ops.convert_to_numpy(scale)
        np.testing.assert_allclose(shift[:3], [0.01984364, 0.10149707, 0.29689495], atol=1e-6)
        np.testing.assert_allclose(scale[:3], [1.63933691, 1.70204478, 1.73642566], atol=1e-6)
        # also tail values
        np.testing.assert_allclose(shift[-1], -0.01760592, atol=1e-6)
        np.testing.assert_allclose(scale[-1], 1.68533454, atol=1e-6)


class TestPresetsBuild:
    @pytest.mark.parametrize("variant", ["tiny", "full"])
    def test_preset_builds(self, variant):
        config, ae = get_ideogram4_config(variant)
        assert isinstance(config, Ideogram4Config)
        assert isinstance(ae, AutoEncoderParams)

    @pytest.mark.parametrize("variant", ["tiny", "full"])
    def test_preset_invariants_hold(self, variant):
        config, ae = get_ideogram4_config(variant)
        # head_dim integer + even
        assert config.emb_dim % config.num_heads == 0
        head_dim = config.head_dim
        assert head_dim % 2 == 0
        # mRoPE band fits
        half = head_dim // 2
        for axis, offset in ((1, 1), (2, 2)):
            consumed = np.arange(offset, config.mrope_section[axis] * 3, 3)
            if consumed.size:
                assert consumed.max() < half
        # VAE channels /32
        assert ae.ch % 32 == 0
        for m in ae.ch_mult:
            assert (ae.ch * m) % 32 == 0
        # latent linkage
        assert config.in_channels == ae.z_channels * config.patch_size ** 2
        assert config.z_channels == ae.z_channels

    def test_unknown_variant_raises(self):
        with pytest.raises(ValueError):
            get_ideogram4_config("nonexistent")


class TestConfigInvariants:
    def test_head_dim_non_divisible_raises(self):
        bad = dict(_TINY, emb_dim=130, num_heads=4)  # 130 % 4 != 0
        with pytest.raises(ValueError, match="divisible by num_heads"):
            Ideogram4Config(**bad)

    def test_odd_head_dim_raises(self):
        # emb_dim=12, num_heads=4 -> head_dim=3 (odd). Keep in_channels linkage valid.
        bad = dict(_TINY, emb_dim=12, num_heads=4)
        with pytest.raises(ValueError, match="must be even"):
            Ideogram4Config(**bad)

    def test_mrope_band_out_of_range_raises(self):
        # head_dim=32 -> half=16; w band 6 -> arange(2, 18, 3).max()=17 >= 16.
        bad = dict(_TINY, mrope_section=(4, 3, 6))
        with pytest.raises(ValueError, match="exceeds"):
            Ideogram4Config(**bad)

    def test_mrope_wrong_length_raises(self):
        bad = dict(_TINY, mrope_section=(4, 3))
        with pytest.raises(ValueError, match="length 3"):
            Ideogram4Config(**bad)

    def test_in_channels_mismatch_raises(self):
        # z=8, patch=2 -> expected in_channels=32; provide 31.
        bad = dict(_TINY, in_channels=31)
        with pytest.raises(ValueError, match="z_channels \\* patch_size"):
            Ideogram4Config(**bad)

    def test_vae_channel_not_div_32_raises(self):
        # Build a config whose paired AE has a bad ch; go through get_ideogram4_config
        # is preset-only, so test the validator directly via a crafted AE.
        from dl_techniques.models.ideogram4.config import _validate_vae_groupnorm

        # base ch not divisible by 32 -> caught on the base-ch guard.
        ae_bad_ch = AutoEncoderParams(ch=48, ch_mult=(1, 2), z_channels=8)
        with pytest.raises(ValueError, match="divisible by 32"):
            _validate_vae_groupnorm(ae_bad_ch)

        # another non-/32 base.
        ae_break = AutoEncoderParams(ch=16, ch_mult=(1, 2), z_channels=8)
        with pytest.raises(ValueError, match="divisible by 32"):
            _validate_vae_groupnorm(ae_break)

        # valid AE passes (sanity that the guard is not over-eager).
        _validate_vae_groupnorm(AutoEncoderParams(ch=32, ch_mult=(1, 2), z_channels=8))


class TestRoundTrip:
    @pytest.mark.parametrize("variant", ["tiny", "full"])
    def test_config_round_trip(self, variant):
        config, _ = get_ideogram4_config(variant)
        d = config.to_dict()
        # mrope_section must serialize as a list
        assert isinstance(d["mrope_section"], list)
        restored = Ideogram4Config.from_dict(d)
        assert restored == config
        assert isinstance(restored.mrope_section, tuple)

    @pytest.mark.parametrize("variant", ["tiny", "full"])
    def test_ae_round_trip(self, variant):
        _, ae = get_ideogram4_config(variant)
        d = ae.to_dict()
        assert isinstance(d["ch_mult"], list)
        restored = AutoEncoderParams.from_dict(d)
        assert restored == ae
        assert isinstance(restored.ch_mult, tuple)

    def test_config_is_frozen(self):
        config, _ = get_ideogram4_config("tiny")
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.emb_dim = 999  # type: ignore[misc]
