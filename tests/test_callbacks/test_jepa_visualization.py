"""Unit tests for jepa_visualization callbacks.

Covers:
- Files get written at the expected frequency.
- Callback does not perturb model trainable weights.
- Graceful degradation when ``mask_prediction_enabled=False``.
- Frequency gating skips intermediate epochs.

Plan plan-2026-11-11T131349-13b28b37 step 17 (D-020) adds: both callbacks fire after
epochs ``frequency``, ``2 * frequency``, ... (1-based, like every sibling) and name the
file by the completed epoch; both draw ONE fixed mask (the same one) from a per-callback
``mask_seed`` and leave the global Keras seed generator alone; ``_disabled`` is reset per
``fit``; a standardized frame is shown with its full range.
"""

from __future__ import annotations

import os

# Ensure matplotlib uses a non-interactive backend before any import.
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pytest

import keras

from dl_techniques.models.world_models.video_jepa.config import VideoJEPAConfig
from dl_techniques.models.world_models.video_jepa.model import VideoJEPA
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
        dropout_rate=0.0,
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
        dropout_rate=0.0,
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
        out = tmp_path / "epoch_001_mask_overlay.png"
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
        cb.on_epoch_end(epoch=0, logs={})  # (0 + 1) % 2 != 0 → skip
        assert list(tmp_path.glob("*.png")) == [], (
            "frequency-gated callback unexpectedly wrote a PNG"
        )
        cb.on_epoch_end(epoch=1, logs={})  # (1 + 1) % 2 == 0 → write
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
        out = tmp_path / "epoch_001_patch_error.png"
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
        cb.on_epoch_end(epoch=0, logs={})
        assert list(tmp_path.glob("*.png")) == []
        cb.on_epoch_end(epoch=1, logs={})
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
        out = tmp_path / "epoch_001_patch_error.png"
        assert out.exists()
        assert out.stat().st_size > 0


# ----------------------------------------------------------------------
# Plan plan-2026-11-11T131349-13b28b37 step 17 (D-020)
# ----------------------------------------------------------------------

import inspect

from dl_techniques.callbacks.jepa_visualization import _to_display_image


def _global_seed_state() -> np.ndarray:
    """The state of the GLOBAL Keras seed generator (``set_random_seed`` does not reset it)."""
    return np.array(keras.src.random.seed_generator.global_seed_generator().state)


def _capture_masks(cb) -> list:
    """Replace the callback's ``_save_grid`` with one that records the mask it was given."""
    captured: list = []
    original = cb._save_grid

    if isinstance(cb, LatentMaskOverlayCallback):
        def record(epoch, mask_np):
            captured.append(mask_np.copy())
    else:
        def record(epoch, err_np, mask_np):
            captured.append(mask_np.copy())

    cb._save_grid = record
    cb._original_save_grid = original
    return captured


def _make(kind, pixels, out, **kwargs):
    cls = LatentMaskOverlayCallback if kind == "overlay" else PatchPredictionErrorCallback
    return cls(eval_pixels=pixels, output_dir=str(out), **kwargs)


@pytest.mark.parametrize("kind", ["overlay", "error"])
class TestMasksAndRandomState:

    # DECISION plan-2026-11-11T131349-13b28b37/D-020 guard: a diagnostic must not move the run's RNG
    def test_the_global_seed_generator_is_untouched(
        self, kind, tiny_cfg_enabled, tmp_path
    ):
        model, pixels = _build_model_and_batch(tiny_cfg_enabled)
        cb = _make(kind, pixels, tmp_path)
        cb.set_model(model)
        cb.on_train_begin()
        before = _global_seed_state()
        for epoch in range(3):
            cb.on_epoch_end(epoch, {})
        np.testing.assert_array_equal(_global_seed_state(), before)
        assert len(list(tmp_path.glob("*.png"))) == 3, "the callback must still draw"

    def test_the_mask_is_the_same_every_epoch(
        self, kind, tiny_cfg_enabled, tmp_path
    ):
        model, pixels = _build_model_and_batch(tiny_cfg_enabled)
        cb = _make(kind, pixels, tmp_path)
        cb.set_model(model)
        cb.on_train_begin()
        masks = _capture_masks(cb)
        for epoch in range(3):
            cb.on_epoch_end(epoch, {})
        assert len(masks) == 3
        np.testing.assert_array_equal(masks[0], masks[1])
        np.testing.assert_array_equal(masks[0], masks[2])

    def test_mask_seed_selects_the_mask(self, kind, tiny_cfg_enabled, tmp_path):
        model, pixels = _build_model_and_batch(tiny_cfg_enabled)
        drawn = {}
        for seed in (1, 2):
            cb = _make(kind, pixels, tmp_path, mask_seed=seed)
            cb.set_model(model)
            cb.on_train_begin()
            masks = _capture_masks(cb)
            cb.on_epoch_end(0, {})
            drawn[seed] = masks[0]
        assert not np.array_equal(drawn[1], drawn[2])


class TestBothCallbacksShowTheSameMask:

    def test_the_default_mask_seed_is_shared(self):
        defaults = {
            cls: inspect.signature(cls.__init__).parameters["mask_seed"].default
            for cls in (LatentMaskOverlayCallback, PatchPredictionErrorCallback)
        }
        assert len(set(defaults.values())) == 1, defaults

    def test_overlay_and_error_draw_the_identical_mask(
        self, tiny_cfg_enabled, tmp_path
    ):
        model, pixels = _build_model_and_batch(tiny_cfg_enabled)
        overlay = _make("overlay", pixels, tmp_path / "a")
        error = _make("error", pixels, tmp_path / "b")
        drawn = []
        for cb in (overlay, error):
            cb.set_model(model)
            cb.on_train_begin()
            drawn.append(_capture_masks(cb))
            cb.on_epoch_end(0, {})
        np.testing.assert_array_equal(drawn[0][0], drawn[1][0])


class TestCadenceAndFiles:

    @pytest.mark.parametrize("kind,suffix", [
        ("overlay", "mask_overlay"), ("error", "patch_error"),
    ])
    def test_frequency_five_fires_after_epochs_five_and_ten(
        self, kind, suffix, tiny_cfg_enabled, tmp_path
    ):
        model, pixels = _build_model_and_batch(tiny_cfg_enabled)
        cb = _make(kind, pixels, tmp_path, frequency=5, max_samples=1, max_frames=1)
        cb.set_model(model)
        cb.on_train_begin()
        for epoch in range(10):
            cb.on_epoch_end(epoch, {})
        assert sorted(p.name for p in tmp_path.glob("*.png")) == [
            f"epoch_005_{suffix}.png", f"epoch_010_{suffix}.png",
        ]


class TestDisabledIsPerRun:

    @pytest.mark.parametrize("kind", ["overlay", "error"])
    def test_a_disabled_first_run_does_not_silence_the_next(
        self, kind, tiny_cfg_enabled, tiny_cfg_disabled, tmp_path
    ):
        cb = _make(kind, np.zeros((2, 2, 32, 32, 3), "float32"), tmp_path)
        # overlay: masking switched off disables it; error heatmap: no config disables it
        first = (
            _build_model_and_batch(tiny_cfg_disabled)[0] if kind == "overlay"
            else type("NoConfig", (), {})()
        )
        cb.set_model(first)
        cb.on_train_begin()
        assert cb._disabled is True
        model, pixels = _build_model_and_batch(tiny_cfg_enabled)
        cb.eval_pixels = pixels
        cb.set_model(model)
        cb.on_train_begin()
        assert cb._disabled is False
        cb.on_epoch_end(0, {})
        assert len(list(tmp_path.glob("*.png"))) == 1


class TestDisplayImage:

    # DECISION plan-2026-11-11T131349-13b28b37/D-020 guard: a standardized frame is not divided by 255
    def test_a_standardized_frame_keeps_its_range(self):
        rng = np.random.default_rng(0)
        frame = rng.standard_normal((16, 16, 3)).astype(np.float32)
        assert frame.min() < -1.5 and frame.max() > 1.5
        shown = _to_display_image(frame)
        assert shown.max() >= 0.5
        assert shown.min() == pytest.approx(0.0, abs=1e-6)
        assert shown.max() == pytest.approx(1.0, abs=1e-6)
        # order preserved: min-max is monotone
        order = np.argsort(frame.ravel())
        assert np.all(np.diff(shown.ravel()[order]) >= -1e-7)

    def test_a_constant_negative_frame_does_not_divide_by_zero(self):
        frame = np.full((4, 4, 3), -3.0, dtype=np.float32)
        shown = _to_display_image(frame)
        assert np.isfinite(shown).all()

    def test_the_known_conventions_are_unchanged(self):
        unit = np.linspace(0, 1, 48, dtype=np.float32).reshape(4, 4, 3)
        np.testing.assert_allclose(_to_display_image(unit), unit)
        signed = np.linspace(-1, 1, 48, dtype=np.float32).reshape(4, 4, 3)
        np.testing.assert_allclose(_to_display_image(signed), (signed + 1) / 2, atol=1e-6)
        byte = np.linspace(0, 255, 48, dtype=np.float32).reshape(4, 4, 3)
        np.testing.assert_allclose(_to_display_image(byte), byte / 255.0, atol=1e-6)
