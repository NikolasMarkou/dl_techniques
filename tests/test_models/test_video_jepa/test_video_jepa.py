"""Video-JEPA-Clifford test suite.

Hardest-first order per decisions.md:

1. Causality (C1)
2. SIGReg stability (C2)
3. AdaLN-zero identity-at-init (C3)
4. Serialization round-trip (C4)
5. Shapes (C5)
6. Streaming O(1) (C6)

At step-1 only :class:`TestConfig` is implemented; subsequent steps append
the remaining test classes.
"""

from __future__ import annotations

import pytest

from dl_techniques.models.video_jepa.config import VideoJEPAConfig


class TestConfig:
    """Validate :class:`VideoJEPAConfig` invariants and round-trip."""

    def test_defaults_construct(self) -> None:
        cfg = VideoJEPAConfig()
        assert cfg.img_size == 64
        assert cfg.patch_size == 8
        assert cfg.patches_per_side == 8
        assert cfg.num_patches == 64
        assert cfg.embed_dim == cfg.cond_dim
        assert cfg.input_image_shape == (64, 64, 3)

    def test_to_from_dict_round_trip(self) -> None:
        cfg = VideoJEPAConfig()
        d = cfg.to_dict()
        # tuples must survive as lists through dict → from_dict and come
        # back as tuples on the reconstructed config.
        assert isinstance(d["encoder_shifts"], list)
        assert isinstance(d["predictor_shifts"], list)
        cfg2 = VideoJEPAConfig.from_dict(d)
        assert cfg2 == cfg
        assert isinstance(cfg2.encoder_shifts, tuple)
        assert isinstance(cfg2.predictor_shifts, tuple)

    def test_custom_fields_survive(self) -> None:
        cfg = VideoJEPAConfig(
            img_size=32, patch_size=8, embed_dim=32, cond_dim=32,
            num_frames=3, history_size_k=3, predictor_depth=1,
            encoder_clifford_depth=1, telemetry_dim=5, sigreg_num_proj=8,
        )
        cfg2 = VideoJEPAConfig.from_dict(cfg.to_dict())
        assert cfg2.img_size == 32
        assert cfg2.patch_size == 8
        assert cfg2.num_frames == 3
        assert cfg2.telemetry_dim == 5
        assert cfg2.sigreg_num_proj == 8

    def test_img_size_divisible_by_patch_size(self) -> None:
        with pytest.raises(ValueError, match="divisible by patch_size"):
            VideoJEPAConfig(img_size=65, patch_size=8)

    def test_cond_dim_must_equal_embed_dim(self) -> None:
        with pytest.raises(ValueError, match="cond_dim .* must equal embed_dim"):
            VideoJEPAConfig(embed_dim=64, cond_dim=32)

    def test_positive_integer_guards(self) -> None:
        with pytest.raises(ValueError, match="history_size_k"):
            VideoJEPAConfig(history_size_k=0)
        with pytest.raises(ValueError, match="num_frames"):
            VideoJEPAConfig(num_frames=0)
        with pytest.raises(ValueError, match="encoder_clifford_depth"):
            VideoJEPAConfig(encoder_clifford_depth=0)
        with pytest.raises(ValueError, match="predictor_depth"):
            VideoJEPAConfig(predictor_depth=0)
