"""Tests for ``models.sd3_mmdit.config``.

Covers the frozen-dataclass invariants, the PRESETS factory, dict round-trip,
and the tiny-VAE GroupNorm divisibility.
"""

import pytest

from dl_techniques.models.ideogram4.config import AutoEncoderParams
from dl_techniques.models.sd3_mmdit.config import (
    SD3MMDiTConfig,
    PRESETS,
    get_sd3_config,
)


class TestGetSD3Config:
    """Factory returns a valid ``(config, ae)`` pair per preset."""

    @pytest.mark.parametrize("variant", ["tiny", "full"])
    def test_returns_pair(self, variant):
        config, ae = get_sd3_config(variant)
        assert isinstance(config, SD3MMDiTConfig)
        assert isinstance(ae, AutoEncoderParams)
        # SD3 latent is 16-channel; transformer in/out match it.
        assert config.in_channels == 16
        assert config.out_channels == 16
        assert ae.z_channels == 16
        assert config.in_channels == ae.z_channels
        # head_dim integer.
        assert config.embedding_size % config.num_heads == 0

    def test_tiny_key_fields(self):
        config, ae = get_sd3_config("tiny")
        assert config.embedding_size == 192
        assert config.num_heads == 6
        assert config.depth == 4
        assert config.dual_attention_layers == (0,)
        assert config.joint_attention_dim == 512
        assert config.pooled_projection_dim == 256

    def test_full_key_fields(self):
        config, ae = get_sd3_config("full")
        assert config.embedding_size == 1536
        assert config.num_heads == 24
        assert config.depth == 24
        assert config.dual_attention_layers == tuple(range(13))
        assert ae.ch == 128
        assert ae.ch_mult == (1, 2, 4, 4)

    def test_unknown_variant_raises(self):
        with pytest.raises(ValueError, match="Unknown variant"):
            get_sd3_config("does_not_exist")


class TestInvariants:
    """``__post_init__`` rejects malformed configs."""

    def test_embedding_not_divisible_by_heads(self):
        with pytest.raises(ValueError, match="divisible by"):
            SD3MMDiTConfig(embedding_size=192, num_heads=5)

    def test_dual_attention_index_out_of_range(self):
        with pytest.raises(ValueError, match="out of range"):
            SD3MMDiTConfig(depth=4, dual_attention_layers=(4,))

    def test_patch_size_too_small(self):
        with pytest.raises(ValueError, match="patch_size"):
            SD3MMDiTConfig(patch_size=0)

    def test_pos_embed_too_small(self):
        # sample_size // patch_size = 64 // 2 = 32 > pos_embed_max_size=16.
        with pytest.raises(ValueError, match="must cover"):
            SD3MMDiTConfig(sample_size=64, patch_size=2, pos_embed_max_size=16)

    def test_in_out_channel_mismatch(self):
        with pytest.raises(ValueError, match="must equal out_channels"):
            SD3MMDiTConfig(in_channels=16, out_channels=8)


class TestSerialization:
    """``to_dict`` / ``from_dict`` round-trip preserves all fields."""

    @pytest.mark.parametrize("variant", ["tiny", "full"])
    def test_round_trip(self, variant):
        config, _ = get_sd3_config(variant)
        d = config.to_dict()
        # dual_attention_layers serialized as a list.
        assert isinstance(d["dual_attention_layers"], list)
        restored = SD3MMDiTConfig.from_dict(d)
        assert restored == config
        # tuple restored.
        assert isinstance(restored.dual_attention_layers, tuple)
        assert restored.dual_attention_layers == config.dual_attention_layers


class TestTinyVAEGroupNorm:
    """Tiny VAE stage channels satisfy GroupNorm32 divisibility."""

    def test_tiny_vae_divisible_by_32(self):
        _, ae = get_sd3_config("tiny")
        assert ae.ch % 32 == 0
        for m in ae.ch_mult:
            assert (ae.ch * m) % 32 == 0
