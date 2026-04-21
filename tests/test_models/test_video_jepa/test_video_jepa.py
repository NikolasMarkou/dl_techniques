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

import os
import tempfile

import numpy as np
import pytest
import keras

from dl_techniques.models.video_jepa.config import VideoJEPAConfig
from dl_techniques.models.video_jepa.encoder import VideoJEPACliffordEncoder
from dl_techniques.models.video_jepa.telemetry_embedder import TelemetryEmbedder


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


# ============================================================================
# TestEncoder — hybrid PatchEmbedding2D + PE2D + CliffordNetBlock stack (C5)
# ============================================================================


def _default_encoder_kwargs() -> dict:
    return dict(
        embed_dim=32, patch_size=8, img_size=32, img_channels=3,
        depth=1, shifts=(1, 2), dropout=0.0,
    )


class TestEncoder:
    """Shape + serialization tests for :class:`VideoJEPACliffordEncoder`."""

    def test_forward_shape(self) -> None:
        enc = VideoJEPACliffordEncoder(**_default_encoder_kwargs())
        # B_total = B * T = 2 * 2 = 4 (>=2 for BN stability).
        B_total, H, W, C = 4, 32, 32, 3
        x = np.random.rand(B_total, H, W, C).astype("float32")
        y = enc(x, training=False)
        Hp = 32 // 8
        assert tuple(y.shape) == (B_total, Hp, Hp, 32), y.shape

    def test_forward_preserves_finiteness(self) -> None:
        enc = VideoJEPACliffordEncoder(**_default_encoder_kwargs())
        x = np.random.rand(4, 32, 32, 3).astype("float32")
        y = np.asarray(enc(x, training=False))
        assert np.all(np.isfinite(y))

    def test_rejects_bad_rank(self) -> None:
        enc = VideoJEPACliffordEncoder(**_default_encoder_kwargs())
        with pytest.raises(ValueError, match="4D input"):
            # Build via explicit build() with a 3D shape.
            enc.build((4, 32, 3))

    def test_even_embed_dim_enforced(self) -> None:
        with pytest.raises(ValueError, match="even"):
            VideoJEPACliffordEncoder(
                embed_dim=31, patch_size=8, img_size=32, img_channels=3,
            )

    def test_serialization_round_trip(self, tmp_path) -> None:
        kwargs = _default_encoder_kwargs()
        # Wrap in a tiny functional model so save() captures the layer.
        inputs = keras.Input(shape=(32, 32, 3))
        enc = VideoJEPACliffordEncoder(**kwargs, name="enc")
        outputs = enc(inputs)
        model = keras.Model(inputs, outputs, name="enc_wrap")

        x = np.random.rand(4, 32, 32, 3).astype("float32")
        y_before = np.asarray(model(x, training=False))

        path = str(tmp_path / "enc.keras")
        model.save(path)
        del model, enc
        keras.backend.clear_session()
        reloaded = keras.models.load_model(path)
        y_after = np.asarray(reloaded(x, training=False))

        # Round-trip must be numerically equal within atol 1e-5.
        np.testing.assert_allclose(y_after, y_before, atol=1e-5, rtol=1e-5)


# ============================================================================
# TestTelemetryEmbedder — continuous sin/cos + LN + Dense
# ============================================================================


class TestTelemetryEmbedder:
    def test_forward_shape(self) -> None:
        emb = TelemetryEmbedder(cond_dim=32, telemetry_dim=7)
        t = np.random.randn(2, 4, 7).astype("float32")
        y = emb(t, training=False)
        assert tuple(y.shape) == (2, 4, 32), y.shape

    def test_forward_finite(self) -> None:
        emb = TelemetryEmbedder(cond_dim=32, telemetry_dim=7)
        t = np.random.randn(2, 4, 7).astype("float32")
        y = np.asarray(emb(t, training=False))
        assert np.all(np.isfinite(y))

    def test_rejects_bad_ndim(self) -> None:
        emb = TelemetryEmbedder(cond_dim=32, telemetry_dim=7)
        with pytest.raises(ValueError, match="3D input"):
            emb.build((2, 7))
        emb2 = TelemetryEmbedder(cond_dim=32, telemetry_dim=7)
        with pytest.raises(ValueError, match="must equal telemetry_dim"):
            emb2.build((2, 4, 5))

    def test_serialization_round_trip(self, tmp_path) -> None:
        inputs = keras.Input(shape=(4, 7))
        emb = TelemetryEmbedder(cond_dim=32, telemetry_dim=7, name="tel_emb")
        out = emb(inputs)
        model = keras.Model(inputs, out, name="tel_wrap")
        t = np.random.randn(2, 4, 7).astype("float32")
        y_before = np.asarray(model(t, training=False))
        path = str(tmp_path / "tel.keras")
        model.save(path)
        del model, emb
        keras.backend.clear_session()
        reloaded = keras.models.load_model(path)
        y_after = np.asarray(reloaded(t, training=False))
        np.testing.assert_allclose(y_after, y_before, atol=1e-5, rtol=1e-5)
