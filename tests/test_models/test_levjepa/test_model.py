"""Tests for the LeVJEPA scale table, variant registry and factory."""

import keras
import pytest

from dl_techniques.models.vision.levjepa.encoder import LeVJEPAEncoder
from dl_techniques.models.vision.levjepa.model import (
    SCALE_CONFIGS,
    MODEL_VARIANTS,
    from_variant,
    create_levjepa,
)

# Ported verbatim from the reference's vit_tiny .. vit_gigantic factories,
# re-quoted from the task prompt's table (embed_dim, depth, num_heads,
# mlp_ratio, patch_size).
_EXPECTED_TABLE = {
    "vit_tiny": (192, 12, 3, 4.0, 16),
    "vit_small": (384, 12, 6, 4.0, 16),
    "vit_base": (768, 12, 12, 4.0, 16),
    "vit_large": (1024, 24, 16, 4.0, 16),
    "vit_huge": (1280, 32, 16, 4.0, 16),
    "vit_giant": (1408, 40, 16, 48.0 / 11.0, 16),
    "vit_gigantic": (1664, 48, 16, 64.0 / 13.0, 14),
}


class TestScaleConfigsTable:
    def test_keys_match_the_reference_variant_names(self):
        assert set(SCALE_CONFIGS.keys()) == set(_EXPECTED_TABLE.keys())

    @pytest.mark.parametrize("variant", list(_EXPECTED_TABLE.keys()))
    def test_values_match_the_reference_table_exactly(self, variant):
        assert SCALE_CONFIGS[variant] == _EXPECTED_TABLE[variant]

    def test_model_variants_keys_match_scale_configs(self):
        assert set(MODEL_VARIANTS.keys()) == set(SCALE_CONFIGS.keys())


class TestFromVariant:
    @pytest.mark.parametrize("variant", list(_EXPECTED_TABLE.keys()))
    def test_from_variant_constructs_without_error(self, variant):
        # Tiny spatial size so even vit_gigantic's depth/width builds fast;
        # patch_size varies per variant (14 for gigantic), so use a size
        # divisible by both 14 and 16.
        enc = from_variant(variant, input_shape=(112, 112, 3))
        assert isinstance(enc, LeVJEPAEncoder)
        embed_dim, depth, num_heads, mlp_ratio, patch_size = SCALE_CONFIGS[variant]
        assert enc.embed_dim == embed_dim
        assert enc.depth == depth
        assert enc.num_heads == num_heads
        assert enc.mlp_ratio == pytest.approx(mlp_ratio)
        assert enc.patch_size == patch_size

    def test_unknown_variant_raises(self):
        with pytest.raises(ValueError, match="Unknown variant"):
            from_variant("vit_nonexistent")


class TestCreateLevjepa:
    def test_factory_builds_a_working_encoder(self):
        enc = create_levjepa(variant="vit_tiny", input_shape=(32, 32, 3))
        x = keras.random.normal((1, 32, 32, 3))
        out = enc(x)
        assert out.shape[0] == 1
        assert out.shape[-1] == 192

    def test_factory_forwards_video_kwargs(self):
        enc = create_levjepa(
            variant="vit_tiny",
            input_shape=(32, 32, 3),
            num_frames=4,
            tubelet_size=2,
            use_rope=True,
            attn_mode="block_causal",
        )
        x = keras.random.normal((1, 4, 32, 32, 3))
        out = enc(x)
        assert out.shape[0] == 1
        assert out.shape[-1] == 192
