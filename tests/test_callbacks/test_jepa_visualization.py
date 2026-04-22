"""Unit tests for jepa_visualization callbacks.

Covers:
- Files get written at the expected frequency.
- Callback does not perturb model trainable weights.
- Graceful degradation when ``mask_prediction_enabled=False``.
- Frequency gating skips intermediate epochs.
"""

from __future__ import annotations

import os

# Ensure matplotlib uses a non-interactive backend before any import.
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pytest

import keras

from dl_techniques.models.video_jepa.config import VideoJEPAConfig
from dl_techniques.models.video_jepa.model import VideoJEPA
from dl_techniques.callbacks.jepa_visualization import (
    LatentMaskOverlayCallback,
    PatchPredictionErrorCallback,
)


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_cfg_enabled() -> VideoJEPAConfig:
    return VideoJEPAConfig(
        img_size=32,
        img_channels=3,
        patch_size=8,
        embed_dim=16,
        num_frames=2,
        history_size_k=2,
        encoder_clifford_depth=1,
        encoder_shifts=(1,),
        predictor_depth=1,
        predictor_num_heads=2,
        predictor_dim_head=8,
        predictor_mlp_dim=32,
        predictor_shifts=(1,),
        sigreg_knots=5,
        sigreg_num_proj=4,
        sigreg_weight=0.0,
        dropout=0.0,
        mask_prediction_enabled=True,
        mask_ratio=0.5,
        lambda_next_frame=1.0,
        lambda_mask=1.0,
    )


@pytest.fixture(scope="module")
def tiny_cfg_disabled() -> VideoJEPAConfig:
    return VideoJEPAConfig(
        img_size=32,
        img_channels=3,
        patch_size=8,
        embed_dim=16,
        num_frames=2,
        history_size_k=2,
        encoder_clifford_depth=1,
        encoder_shifts=(1,),
        predictor_depth=1,
        predictor_num_heads=2,
        predictor_dim_head=8,
        predictor_mlp_dim=32,
        predictor_shifts=(1,),
        sigreg_knots=5,
        sigreg_num_proj=4,
        sigreg_weight=0.0,
        dropout=0.0,
        mask_prediction_enabled=False,
        mask_ratio=0.5,
        lambda_next_frame=1.0,
        lambda_mask=0.0,
    )


def _build_model_and_batch(cfg: VideoJEPAConfig):
    model = VideoJEPA(config=cfg)
    rng = np.random.default_rng(0)
    B, T = 2, cfg.num_frames
    pixels = rng.standard_normal(
        (B, T, cfg.img_size, cfg.img_size, cfg.img_channels)
    ).astype(np.float32)
    # Force-build the model so encoder/predictor sub-weights exist.
    _ = model({"pixels": pixels}, training=False)
    return model, pixels


def _snapshot_weights(model) -> list[np.ndarray]:
    return [np.asarray(w).copy() for w in model.trainable_weights]


def _weights_unchanged(before, after, atol=0.0) -> bool:
    return all(np.allclose(a, b, atol=atol) for a, b in zip(before, after))


# ----------------------------------------------------------------------
# Tests — LatentMaskOverlayCallback
# ----------------------------------------------------------------------


class TestLatentMaskOverlayCallback:

    def test_writes_png_on_epoch_end(self, tiny_cfg_enabled, tmp_path):
        model, pixels = _build_model_and_batch(tiny_cfg_enabled)
        cb = LatentMaskOverlayCallback(
            eval_pixels=pixels,
            output_dir=str(tmp_path),
            frequency=1,
            max_samples=2,
            max_frames=2,
        )
        cb.set_model(model)
        cb.on_train_begin()
        before = _snapshot_weights(model)

        cb.on_epoch_end(epoch=0, logs={})

        after = _snapshot_weights(model)
        out = tmp_path / "epoch_000_mask_overlay.png"
        assert out.exists(), "expected mask overlay PNG not written"
        assert out.stat().st_size > 0, "mask overlay PNG is empty"
        assert _weights_unchanged(before, after), (
            "LatentMaskOverlayCallback perturbed model weights"
        )

    def test_frequency_gating_skips(self, tiny_cfg_enabled, tmp_path):
        model, pixels = _build_model_and_batch(tiny_cfg_enabled)
        cb = LatentMaskOverlayCallback(
            eval_pixels=pixels,
            output_dir=str(tmp_path),
            frequency=2,
        )
        cb.set_model(model)
        cb.on_train_begin()
        cb.on_epoch_end(epoch=1, logs={})  # 1 % 2 != 0 → skip
        assert list(tmp_path.glob("*.png")) == [], (
            "frequency-gated callback unexpectedly wrote a PNG"
        )
        cb.on_epoch_end(epoch=2, logs={})  # 2 % 2 == 0 → write
        assert (tmp_path / "epoch_002_mask_overlay.png").exists()

    def test_disabled_when_mask_prediction_off(
        self, tiny_cfg_disabled, tmp_path,
    ):
        model, pixels = _build_model_and_batch(tiny_cfg_disabled)
        cb = LatentMaskOverlayCallback(
            eval_pixels=pixels,
            output_dir=str(tmp_path),
            frequency=1,
        )
        cb.set_model(model)
        cb.on_train_begin()
        cb.on_epoch_end(epoch=0, logs={})
        # Disabled path: no crash, no file.
        assert cb._disabled is True
        assert list(tmp_path.glob("*.png")) == []


# ----------------------------------------------------------------------
# Tests — PatchPredictionErrorCallback
# ----------------------------------------------------------------------


class TestPatchPredictionErrorCallback:

    def test_writes_png_on_epoch_end(self, tiny_cfg_enabled, tmp_path):
        model, pixels = _build_model_and_batch(tiny_cfg_enabled)
        cb = PatchPredictionErrorCallback(
            eval_pixels=pixels,
            output_dir=str(tmp_path),
            frequency=1,
            max_samples=2,
            max_frames=2,
        )
        cb.set_model(model)
        cb.on_train_begin()
        before = _snapshot_weights(model)

        cb.on_epoch_end(epoch=0, logs={})

        after = _snapshot_weights(model)
        out = tmp_path / "epoch_000_patch_error.png"
        assert out.exists(), "expected patch error PNG not written"
        assert out.stat().st_size > 0, "patch error PNG is empty"
        assert _weights_unchanged(before, after), (
            "PatchPredictionErrorCallback perturbed model weights"
        )

    def test_frequency_gating_skips(self, tiny_cfg_enabled, tmp_path):
        model, pixels = _build_model_and_batch(tiny_cfg_enabled)
        cb = PatchPredictionErrorCallback(
            eval_pixels=pixels,
            output_dir=str(tmp_path),
            frequency=2,
        )
        cb.set_model(model)
        cb.on_train_begin()
        cb.on_epoch_end(epoch=1, logs={})
        assert list(tmp_path.glob("*.png")) == []
        cb.on_epoch_end(epoch=2, logs={})
        assert (tmp_path / "epoch_002_patch_error.png").exists()

    def test_mask_prediction_off_still_runs(
        self, tiny_cfg_disabled, tmp_path,
    ):
        model, pixels = _build_model_and_batch(tiny_cfg_disabled)
        cb = PatchPredictionErrorCallback(
            eval_pixels=pixels,
            output_dir=str(tmp_path),
            frequency=1,
        )
        cb.set_model(model)
        cb.on_train_begin()
        cb.on_epoch_end(epoch=0, logs={})
        # In the disabled case, prediction error is still meaningful
        # (predictor on unmasked z) — the callback should write a PNG
        # and not touch the model weights.
        out = tmp_path / "epoch_000_patch_error.png"
        assert out.exists()
        assert out.stat().st_size > 0
