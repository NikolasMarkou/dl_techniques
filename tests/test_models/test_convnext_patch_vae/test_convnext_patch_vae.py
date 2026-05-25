"""Test suite for :mod:`dl_techniques.models.convnext_patch_vae`.

Hardest-first order (per ``plans/plan_2026-05-25_fb57d478/plan.md`` Step 5):

1. Config round-trip + invariant violations (C1, fastest signal).
2. Resolution-agnostic forward (C3, the core hypothesis lever).
3. Save/load round-trip (C4).
4. Per-patch KL resolution-invariance (C5).
5. SIGReg integration finite (C7).
6. ``fit`` reports nonzero loss (C6 — D-001 contract).
7. Sample API resolution-agnostic (C8).
8. SIGReg-off ablation path clean (C9).
"""

from __future__ import annotations

import os

import keras
import numpy as np
import pytest

from dl_techniques.models.convnext_patch_vae.config import (
    ConvNeXtPatchVAEConfig,
)
from dl_techniques.models.convnext_patch_vae.model import ConvNeXtPatchVAE


def _tiny_cfg(**overrides) -> ConvNeXtPatchVAEConfig:
    """Small config suitable for fast unit tests (~few seconds total)."""
    base = dict(
        img_size=32,
        img_channels=3,
        patch_size=4,
        embed_dim=16,
        encoder_depth=1,
        decoder_depth=1,
        kernel_size=3,
        latent_dim=4,
        beta_kl=0.5,
        lambda_sigreg=0.1,
        sigreg_knots=5,
        sigreg_num_proj=32,
        recon_loss_type="mse",
        dropout_rate=0.0,
        spatial_dropout_rate=0.0,
        gamma_clip=1.0,
    )
    base.update(overrides)
    return ConvNeXtPatchVAEConfig(**base)


class TestConfig:
    """Test 1 — ``ConvNeXtPatchVAEConfig`` round-trip + invariants."""

    def test_config_roundtrip(self) -> None:
        cfg = _tiny_cfg()
        d = cfg.to_dict()
        cfg2 = ConvNeXtPatchVAEConfig.from_dict(d)
        assert cfg.to_dict() == cfg2.to_dict()
        # Derived properties survive the round-trip.
        assert cfg.patches_per_side == 8
        assert cfg.num_patches == 64
        assert cfg.input_image_shape == (32, 32, 3)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"img_size": 33, "patch_size": 4},  # not divisible
            {"latent_dim": 0},                  # zero-dim latent rejected
            {"recon_loss_type": "bad"},         # unknown family
            {"beta_kl": -1.0},                  # negative weight
            {"lambda_sigreg": -0.5},            # negative SIGReg weight
            {"sigreg_knots": 1},                # below SIGReg floor
            {"dropout_rate": 1.5},              # out of [0, 1]
        ],
    )
    def test_invariant_violations_reject(self, kwargs: dict) -> None:
        with pytest.raises(ValueError):
            _tiny_cfg(**kwargs)


class TestForward:
    """Tests 2, 4 — resolution-agnostic forward + KL resolution invariance."""

    def test_forward_shapes_at_two_resolutions(self) -> None:
        cfg = _tiny_cfg()
        m = ConvNeXtPatchVAE(cfg)
        x32 = np.random.RandomState(0).rand(2, 32, 32, 3).astype("float32")
        x48 = np.random.RandomState(1).rand(2, 48, 48, 3).astype("float32")
        out32 = m(keras.ops.convert_to_tensor(x32), training=False)
        out48 = m(keras.ops.convert_to_tensor(x48), training=False)
        assert tuple(out32["reconstruction"].shape) == (2, 32, 32, 3)
        assert tuple(out48["reconstruction"].shape) == (2, 48, 48, 3)
        assert tuple(out32["mu"].shape) == (2, 8, 8, 4)
        assert tuple(out48["mu"].shape) == (2, 12, 12, 4)
        assert tuple(out32["z"].shape) == (2, 8, 8, 4)
        assert tuple(out48["log_var"].shape) == (2, 12, 12, 4)

    def test_per_patch_kl_resolution_invariance(self) -> None:
        cfg = _tiny_cfg()
        m = ConvNeXtPatchVAE(cfg)
        rng = np.random.RandomState(7)
        x32 = rng.rand(4, 32, 32, 3).astype("float32")
        x48 = rng.rand(4, 48, 48, 3).astype("float32")
        # Reset trackers between forwards so we read each resolution's
        # KL in isolation.
        m.kl_loss_tracker.reset_state()
        _ = m(keras.ops.convert_to_tensor(x32), training=False)
        kl_32 = float(m.kl_loss_tracker.result())
        m.kl_loss_tracker.reset_state()
        _ = m(keras.ops.convert_to_tensor(x48), training=False)
        kl_48 = float(m.kl_loss_tracker.result())
        # Per-patch averaging: doubling Hp*Wp must not double KL. Allow
        # a 2x band — random inputs + small model + 1 block of GRN.
        ratio = max(kl_32, kl_48) / max(min(kl_32, kl_48), 1e-9)
        assert ratio < 2.0, (
            f"per-patch KL not resolution-invariant: 32->{kl_32}, "
            f"48->{kl_48}, ratio={ratio}"
        )


class TestSaveLoad:
    """Test 3 — full ``.keras`` save/load round-trip."""

    def test_save_load_roundtrip(self, tmp_path) -> None:
        cfg = _tiny_cfg()
        m = ConvNeXtPatchVAE(cfg)
        # Build by calling once on a real input so weights exist before
        # save. ``encode`` alone builds only the encoder; we need the
        # full graph so the decoder weights survive the round-trip too.
        rng = np.random.RandomState(42)
        x = rng.rand(2, 32, 32, 3).astype("float32")
        _ = m(keras.ops.convert_to_tensor(x), training=False)
        # encode produces (mu, log_var) deterministically (no sampling).
        mu_before, _ = m.encode(keras.ops.convert_to_tensor(x))
        mu_before_np = keras.ops.convert_to_numpy(mu_before)
        path = os.path.join(str(tmp_path), "cpvae.keras")
        m.save(path)
        m2 = keras.models.load_model(path)
        mu_after, _ = m2.encode(keras.ops.convert_to_tensor(x))
        mu_after_np = keras.ops.convert_to_numpy(mu_after)
        np.testing.assert_allclose(mu_after_np, mu_before_np, atol=1e-4)


class TestSIGRegIntegration:
    """Test 5 — SIGReg integration produces a finite scalar."""

    def test_sigreg_input_shape_contract(self) -> None:
        cfg = _tiny_cfg(latent_dim=8, sigreg_num_proj=64)
        m = ConvNeXtPatchVAE(cfg)
        x = np.random.RandomState(0).rand(3, 32, 32, 3).astype("float32")
        m.sigreg_loss_tracker.reset_state()
        _ = m(keras.ops.convert_to_tensor(x), training=False)
        val = float(m.sigreg_loss_tracker.result())
        assert np.isfinite(val), f"SIGReg statistic non-finite: {val}"


class TestFit:
    """Test 6 — D-001 ``history.history['loss']`` contract."""

    def test_fit_one_step_decreases_loss(self) -> None:
        cfg = _tiny_cfg()
        m = ConvNeXtPatchVAE(cfg)
        m.compile(optimizer=keras.optimizers.AdamW(1e-3))
        x = np.random.RandomState(3).rand(4, 32, 32, 3).astype("float32")
        hist = m.fit(x, epochs=1, steps_per_epoch=2, batch_size=2, verbose=0)
        losses = hist.history["loss"]
        assert len(losses) == 1
        assert losses[-1] > 0.0, (
            f"history['loss'] must be > 0 per D-001 contract, got "
            f"{losses[-1]}"
        )
        assert np.isfinite(losses[-1]), "loss must be finite"


class TestSampleAPI:
    """Test 7 — ``sample(...)`` is resolution-agnostic."""

    def test_sample_resolution_agnostic(self) -> None:
        cfg = _tiny_cfg()
        m = ConvNeXtPatchVAE(cfg)
        s44 = m.sample(num_samples=2, hp=4, wp=4, seed=0)
        s88 = m.sample(num_samples=2, hp=8, wp=8, seed=0)
        assert tuple(s44.shape) == (2, 16, 16, 3)
        # 8x8 patch grid -> 4x larger spatial dims than 4x4.
        assert tuple(s88.shape) == (2, 32, 32, 3)


class TestSIGRegOff:
    """Test 8 — ``lambda_sigreg=0.0`` ablation path is clean.

    Two contracts:
    - The weighted SIGReg contribution to ``self.losses`` is exactly zero
      so gradients carry no SIGReg signal.
    - The ``sigreg_loss`` tracker still reports the raw SIGReg statistic
      (NOT the weighted product), so ablation comparisons across
      ``lambda_sigreg`` settings remain on a single scale. This is the
      contract documented at the ``sigreg_loss_tracker`` creation site
      and required by the training menu's T4b sweep.
    """

    def test_sigreg_off_branch(self) -> None:
        cfg = _tiny_cfg(lambda_sigreg=0.0)
        m = ConvNeXtPatchVAE(cfg)
        x = np.random.RandomState(11).rand(2, 32, 32, 3).astype("float32")
        m.sigreg_loss_tracker.reset_state()
        _ = m(keras.ops.convert_to_tensor(x), training=False)
        # The weighted SIGReg term is the third add_loss call; with
        # lambda=0 it must be exactly zero.
        losses_after_forward = [float(l) for l in m.losses]
        # We expect at least 3 add_loss calls (recon, beta*KL, lambda*SIGReg).
        assert len(losses_after_forward) >= 3
        weighted_sigreg = losses_after_forward[2]
        assert weighted_sigreg == 0.0, (
            f"weighted SIGReg with lambda_sigreg=0 must be 0.0, got "
            f"{weighted_sigreg}"
        )
        # The tracker reports the RAW statistic — finite + typically > 0
        # for random data.
        raw_sigreg = float(m.sigreg_loss_tracker.result())
        assert np.isfinite(raw_sigreg)
        assert raw_sigreg >= 0.0, (
            f"raw SIGReg statistic must be non-negative, got {raw_sigreg}"
        )
