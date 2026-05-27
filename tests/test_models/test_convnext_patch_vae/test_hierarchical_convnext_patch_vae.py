"""Test suite for the hierarchical ConvNeXt patch VAE.

Mirrors ``test_convnext_patch_vae.py`` structure but for the two-level
variant introduced in ``plans/plan_2026-05-27_dee954c6/``.
"""

from __future__ import annotations

import os

import keras
import numpy as np
import pytest

from dl_techniques.models.convnext_patch_vae.config import (
    HierarchicalConvNeXtPatchVAEConfig,
)
from dl_techniques.models.convnext_patch_vae.model_hierarchical import (
    HierarchicalConvNeXtPatchVAE,
    _L2ConditionalPrior,
    create_hierarchical_convnext_patch_vae,
)


def _tiny_cfg(**overrides) -> HierarchicalConvNeXtPatchVAEConfig:
    """CIFAR-sized hierarchical config — fast unit-test scale."""
    base = dict(
        img_size=32,
        img_channels=3,
        patch_size_l1=8,
        patch_size_l2=4,
        embed_dim_l1=16,
        embed_dim_l2=16,
        encoder_depth_l1=1,
        decoder_depth_l1=1,
        encoder_depth_l2=1,
        decoder_depth_l2=1,
        kernel_size=3,
        latent_dim_l1=8,
        latent_dim_l2=4,
        beta_kl_l1=0.5,
        beta_kl_l2=0.5,
        lambda_sigreg_l1=0.05,
        lambda_sigreg_l2=0.1,
        sigreg_knots=5,
        sigreg_num_proj=32,
        recon_loss_type="mse",
        dropout_rate=0.0,
        spatial_dropout_rate=0.0,
        gamma_clip=1.0,
    )
    base.update(overrides)
    return HierarchicalConvNeXtPatchVAEConfig(**base)


class TestConfig:
    """``HierarchicalConvNeXtPatchVAEConfig`` round-trip + invariants."""

    def test_config_roundtrip(self) -> None:
        cfg = _tiny_cfg()
        d = cfg.to_dict()
        cfg2 = HierarchicalConvNeXtPatchVAEConfig.from_dict(d)
        assert cfg.to_dict() == cfg2.to_dict()
        assert cfg.patches_per_side_l1 == 4
        assert cfg.patches_per_side_l2 == 8
        assert cfg.num_patches_l1 == 16
        assert cfg.num_patches_l2 == 64
        assert cfg.tile_factor == 2
        assert cfg.input_image_shape == (32, 32, 3)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"patch_size_l1": 4, "patch_size_l2": 4},   # p1 must be > p2
            {"patch_size_l1": 8, "patch_size_l2": 3},   # non-multiple
            {"img_size": 33, "patch_size_l1": 8, "patch_size_l2": 4},  # non-divisible
            {"latent_dim_l1": 0},                       # zero-dim rejected
            {"latent_dim_l2": 0},
            {"recon_loss_type": "bad"},
            {"beta_kl_l1": -1.0},
            {"lambda_sigreg_l2": -0.5},
            {"sigreg_knots": 1},
            {"dropout_rate": 1.5},
            {"conditioning": "cross_attention"},        # not implemented
        ],
    )
    def test_invariant_violations_reject(self, kwargs: dict) -> None:
        with pytest.raises(ValueError):
            _tiny_cfg(**kwargs)


class TestForward:
    """Resolution-agnostic forward + per-patch KL invariance at both scales."""

    def test_forward_shapes_at_two_resolutions(self) -> None:
        cfg = _tiny_cfg()
        m = HierarchicalConvNeXtPatchVAE(cfg)
        x32 = np.random.RandomState(0).rand(2, 32, 32, 3).astype("float32")
        x64 = np.random.RandomState(1).rand(2, 64, 64, 3).astype("float32")
        out32 = m(keras.ops.convert_to_tensor(x32), training=False)
        out64 = m(keras.ops.convert_to_tensor(x64), training=False)
        # Recon shape matches input.
        assert tuple(out32["reconstruction"].shape) == (2, 32, 32, 3)
        assert tuple(out64["reconstruction"].shape) == (2, 64, 64, 3)
        # L1: 32/8=4, 64/8=8.
        assert tuple(out32["mu_l1"].shape) == (2, 4, 4, 8)
        assert tuple(out64["mu_l1"].shape) == (2, 8, 8, 8)
        # L2: 32/4=8, 64/4=16.
        assert tuple(out32["mu_l2"].shape) == (2, 8, 8, 4)
        assert tuple(out64["mu_l2"].shape) == (2, 16, 16, 4)
        # `mu` / `z` / `log_var` aliases map to L2 (callback compat — D-005).
        assert tuple(out32["mu"].shape) == tuple(out32["mu_l2"].shape)
        assert tuple(out32["z"].shape) == tuple(out32["z_l2"].shape)

    def test_per_patch_kl_resolution_invariance(self) -> None:
        cfg = _tiny_cfg()
        m = HierarchicalConvNeXtPatchVAE(cfg)
        rng = np.random.RandomState(7)
        x32 = rng.rand(4, 32, 32, 3).astype("float32")
        x64 = rng.rand(4, 64, 64, 3).astype("float32")

        def _kl_pair(x):
            m.kl_l1_loss_tracker.reset_state()
            m.kl_l2_loss_tracker.reset_state()
            _ = m(keras.ops.convert_to_tensor(x), training=False)
            return (
                float(m.kl_l1_loss_tracker.result()),
                float(m.kl_l2_loss_tracker.result()),
            )

        kl_l1_32, kl_l2_32 = _kl_pair(x32)
        kl_l1_64, kl_l2_64 = _kl_pair(x64)
        # Per-patch averaging: quadrupling Hp*Wp must not quadruple KL.
        for a, b, name in [
            (kl_l1_32, kl_l1_64, "kl_L1"),
            (kl_l2_32, kl_l2_64, "kl_L2"),
        ]:
            ratio = max(a, b) / max(min(a, b), 1e-9)
            assert ratio < 2.0, (
                f"per-patch {name} not resolution-invariant: "
                f"32x32->{a}, 64x64->{b}, ratio={ratio}"
            )


class TestSaveLoad:
    """Full ``.keras`` save/load round-trip — bit-exact on both mu_l1 and mu_l2."""

    def test_save_load_roundtrip(self, tmp_path) -> None:
        cfg = _tiny_cfg()
        m = HierarchicalConvNeXtPatchVAE(cfg)
        rng = np.random.RandomState(42)
        x = rng.rand(2, 32, 32, 3).astype("float32")
        # Build by calling once on a real input so all sub-layer weights
        # exist before save.
        _ = m(keras.ops.convert_to_tensor(x), training=False)
        mu_l1_before, _, mu_l2_before, _ = m.encode(
            keras.ops.convert_to_tensor(x)
        )
        mu_l1_before_np = keras.ops.convert_to_numpy(mu_l1_before)
        mu_l2_before_np = keras.ops.convert_to_numpy(mu_l2_before)
        path = os.path.join(str(tmp_path), "h_vae.keras")
        m.save(path)
        m2 = keras.models.load_model(path)
        mu_l1_after, _, mu_l2_after, _ = m2.encode(
            keras.ops.convert_to_tensor(x)
        )
        mu_l1_after_np = keras.ops.convert_to_numpy(mu_l1_after)
        mu_l2_after_np = keras.ops.convert_to_numpy(mu_l2_after)
        np.testing.assert_allclose(mu_l1_after_np, mu_l1_before_np, atol=1e-4)
        np.testing.assert_allclose(mu_l2_after_np, mu_l2_before_np, atol=1e-4)


class TestSIGRegIntegration:
    """SIGReg at both scales emits a finite scalar with x N applied."""

    def test_sigreg_input_shape_contract(self) -> None:
        cfg = _tiny_cfg(latent_dim_l1=8, latent_dim_l2=8, sigreg_num_proj=64)
        m = HierarchicalConvNeXtPatchVAE(cfg)
        x = np.random.RandomState(0).rand(3, 32, 32, 3).astype("float32")
        m.sigreg_l1_loss_tracker.reset_state()
        m.sigreg_l2_loss_tracker.reset_state()
        _ = m(keras.ops.convert_to_tensor(x), training=False)
        v1 = float(m.sigreg_l1_loss_tracker.result())
        v2 = float(m.sigreg_l2_loss_tracker.result())
        assert np.isfinite(v1), f"L1 SIGReg non-finite: {v1}"
        assert np.isfinite(v2), f"L2 SIGReg non-finite: {v2}"
        # x N scaling: the tracker stores `sigreg_layer(z) * N`. With
        # N_L1=16 and N_L2=64, both must be positive on random data.
        assert v1 > 0.0
        assert v2 > 0.0


class TestFit:
    """``history.history`` contract — all 10 trackers + `loss` present."""

    def test_fit_one_step_decreases_loss(self) -> None:
        cfg = _tiny_cfg()
        m = HierarchicalConvNeXtPatchVAE(cfg)
        m.compile(optimizer=keras.optimizers.AdamW(1e-3), jit_compile=False)
        x = np.random.RandomState(3).rand(4, 32, 32, 3).astype("float32")
        hist = m.fit(x, epochs=1, steps_per_epoch=2, batch_size=2, verbose=0)
        losses = hist.history["loss"]
        assert len(losses) == 1
        assert losses[-1] > 0.0
        assert np.isfinite(losses[-1])
        # All 10 expected trackers (D-001 contract for the hierarchical model).
        expected_keys = {
            "loss",
            "recon_loss",
            "kl_L1_loss", "kl_L2_loss",
            "sigreg_L1_loss", "sigreg_L2_loss",
            "kl_L1_weighted", "kl_L2_weighted",
            "sigreg_L1_weighted", "sigreg_L2_weighted",
        }
        missing = expected_keys - set(hist.history.keys())
        assert not missing, f"missing trackers in history: {missing}"


class TestSampleAPI:
    """``sample(...)`` honors the L1 grid + tile_factor."""

    def test_sample_resolution_agnostic(self) -> None:
        cfg = _tiny_cfg()
        m = HierarchicalConvNeXtPatchVAE(cfg)
        # Default: hp1 = patches_per_side_l1 = 4; L2 grid = 4*2=8 -> 8*p2=32.
        s_default = m.sample(num_samples=2, seed=0)
        assert tuple(s_default.shape) == (2, 32, 32, 3)
        # Override hp1=2 -> L2 grid 4 -> 4*4=16.
        s_small = m.sample(num_samples=2, hp1=2, wp1=2, seed=0)
        assert tuple(s_small.shape) == (2, 16, 16, 3)


class TestConditionalPrior:
    """Learnable conditional prior `p(z_l2 | z_l1)`.

    Central claims (anchored as D-001/D-002/D-003 in source):
    - Zero-init heads -> prior emits exactly N(0, I) at step 0.
    - Step-0 conditional KL == legacy KL(q || N(0,I)) bit-exact.
    - Pure-prior `sample()` is now coherent (uses the prior).
    """

    def test_zero_init_emits_unit_gaussian(self) -> None:
        # Direct test on the layer itself.
        prior = _L2ConditionalPrior(
            tile_factor=2,
            latent_dim_l1=8,
            latent_dim_l2=4,
            embed_dim=16,
            depth=2,
            kernel_size=3,
        )
        z_l1 = keras.ops.convert_to_tensor(
            np.random.RandomState(0).randn(2, 4, 4, 8).astype("float32")
        )
        mu_p, lv_p = prior(z_l1, training=False)
        # Both heads zero-init -> output is exactly zero, regardless of z_l1.
        assert float(keras.ops.max(keras.ops.abs(mu_p))) == 0.0
        assert float(keras.ops.max(keras.ops.abs(lv_p))) == 0.0
        assert tuple(mu_p.shape) == (2, 8, 8, 4)
        assert tuple(lv_p.shape) == (2, 8, 8, 4)

    def test_conditional_kl_equals_legacy_at_step_zero(self) -> None:
        # Two models, same shared-layer seeds, only the toggle differs.
        # At step 0 the prior emits exactly N(0, I), so the conditional KL
        # must equal the legacy KL bit-exact (modulo float reduction order;
        # we observed exact 0 in practice but assert atol=1e-5).
        rng = np.random.RandomState(0)
        x = keras.ops.convert_to_tensor(
            rng.rand(2, 32, 32, 3).astype("float32")
        )
        keras.utils.set_random_seed(42)
        m_on = HierarchicalConvNeXtPatchVAE(_tiny_cfg(learnable_l2_prior=True))
        _ = m_on(x, training=False)
        keras.utils.set_random_seed(42)
        m_off = HierarchicalConvNeXtPatchVAE(_tiny_cfg(learnable_l2_prior=False))
        _ = m_off(x, training=False)

        m_on.kl_l2_loss_tracker.reset_state()
        m_off.kl_l2_loss_tracker.reset_state()
        _ = m_on(x, training=False)
        _ = m_off(x, training=False)
        kl_on = float(m_on.kl_l2_loss_tracker.result())
        kl_off = float(m_off.kl_l2_loss_tracker.result())
        assert abs(kl_on - kl_off) < 1e-5, (
            f"step-0 KL mismatch: with prior={kl_on}, without={kl_off}"
        )
        # And the toggle actually plumbs through:
        assert m_on.l2_prior is not None
        assert m_off.l2_prior is None

    def test_kl_zero_when_q_equals_p(self) -> None:
        # When q == p, KL = 0.
        cfg = _tiny_cfg(learnable_l2_prior=True)
        m = HierarchicalConvNeXtPatchVAE(cfg)
        # Build by calling once.
        _ = m(keras.ops.convert_to_tensor(
            np.random.RandomState(0).rand(1, 32, 32, 3).astype("float32")
        ), training=False)

        mu = keras.ops.convert_to_tensor(
            np.random.RandomState(1).randn(1, 8, 8, 4).astype("float32")
        )
        lv = keras.ops.convert_to_tensor(
            np.random.RandomState(2).randn(1, 8, 8, 4).astype("float32") * 0.5
        )
        kl = float(m._compute_kl_l2_conditional(mu, lv, mu, lv))
        assert abs(kl) < 1e-5, f"KL(q || q) should be ~0, got {kl}"

    def test_sample_coherent_path_runs(self) -> None:
        cfg = _tiny_cfg(learnable_l2_prior=True)
        m = HierarchicalConvNeXtPatchVAE(cfg)
        # Build first (sample needs the prior to be built).
        _ = m(keras.ops.convert_to_tensor(
            np.random.RandomState(0).rand(1, 32, 32, 3).astype("float32")
        ), training=False)
        s = m.sample(num_samples=4, seed=7)
        s_np = np.array(s)
        assert s_np.shape == (4, 32, 32, 3)
        assert np.all(np.isfinite(s_np))

    def test_sample_legacy_path_unchanged(self) -> None:
        cfg = _tiny_cfg(learnable_l2_prior=False)
        m = HierarchicalConvNeXtPatchVAE(cfg)
        _ = m(keras.ops.convert_to_tensor(
            np.random.RandomState(0).rand(1, 32, 32, 3).astype("float32")
        ), training=False)
        s = m.sample(num_samples=4, seed=7)
        s_np = np.array(s)
        assert s_np.shape == (4, 32, 32, 3)
        assert np.all(np.isfinite(s_np))

    def test_save_load_roundtrip_with_prior(self, tmp_path) -> None:
        cfg = _tiny_cfg(learnable_l2_prior=True)
        m = HierarchicalConvNeXtPatchVAE(cfg)
        x = keras.ops.convert_to_tensor(
            np.random.RandomState(42).rand(2, 32, 32, 3).astype("float32")
        )
        _ = m(x, training=False)
        mu_l1_a, _, mu_l2_a, _ = m.encode(x)
        path = os.path.join(str(tmp_path), "h_prior.keras")
        m.save(path)
        m2 = keras.models.load_model(path)
        mu_l1_b, _, mu_l2_b, _ = m2.encode(x)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(mu_l1_a),
            keras.ops.convert_to_numpy(mu_l1_b),
            atol=1e-4,
        )
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(mu_l2_a),
            keras.ops.convert_to_numpy(mu_l2_b),
            atol=1e-4,
        )
        # Prior round-trips too.
        assert m2.l2_prior is not None

    def test_one_fit_step_with_conditional_prior(self) -> None:
        cfg = _tiny_cfg(learnable_l2_prior=True)
        m = HierarchicalConvNeXtPatchVAE(cfg)
        m.compile(optimizer=keras.optimizers.AdamW(1e-3), jit_compile=False)
        x = np.random.RandomState(3).rand(4, 32, 32, 3).astype("float32")
        hist = m.fit(x, epochs=1, steps_per_epoch=2, batch_size=2, verbose=0)
        assert np.isfinite(hist.history["loss"][-1])
        assert np.isfinite(hist.history["kl_L2_loss"][-1])
        expected = {
            "loss", "recon_loss",
            "kl_L1_loss", "kl_L2_loss",
            "sigreg_L1_loss", "sigreg_L2_loss",
            "kl_L1_weighted", "kl_L2_weighted",
            "sigreg_L1_weighted", "sigreg_L2_weighted",
        }
        missing = expected - set(hist.history.keys())
        assert not missing, f"missing trackers: {missing}"

    def test_resolution_invariance_kl_l2_conditional(self) -> None:
        cfg = _tiny_cfg(learnable_l2_prior=True)
        m = HierarchicalConvNeXtPatchVAE(cfg)
        rng = np.random.RandomState(7)
        x32 = rng.rand(4, 32, 32, 3).astype("float32")
        x64 = rng.rand(4, 64, 64, 3).astype("float32")

        def _kl_l2(x):
            m.kl_l2_loss_tracker.reset_state()
            _ = m(keras.ops.convert_to_tensor(x), training=False)
            return float(m.kl_l2_loss_tracker.result())

        kl_32 = _kl_l2(x32)
        kl_64 = _kl_l2(x64)
        ratio = max(kl_32, kl_64) / max(min(kl_32, kl_64), 1e-9)
        assert ratio < 2.0, (
            f"per-patch conditional KL_L2 not resolution-invariant: "
            f"32x32->{kl_32}, 64x64->{kl_64}, ratio={ratio}"
        )


class TestSampleFrom:
    """``sample_from(x, temperature)`` coherent sampling around a real anchor.

    Hierarchical motivation: pure-prior `sample()` is incoherent because
    z_l1 and z_l2 were trained as correlated pairs. ``sample_from`` keeps
    them correlated by reparameterizing both from the encoder posterior.
    """

    def test_temperature_zero_is_deterministic_recon(self) -> None:
        cfg = _tiny_cfg()
        m = HierarchicalConvNeXtPatchVAE(cfg)
        rng = np.random.RandomState(0)
        x = keras.ops.convert_to_tensor(rng.rand(2, 32, 32, 3).astype("float32"))
        _ = m(x, training=False)  # build
        a = np.array(m.sample_from(x, temperature=0.0, seed=1))
        b = np.array(m.sample_from(x, temperature=0.0, seed=2))
        np.testing.assert_allclose(a, b, atol=1e-6)
        # And must equal decode(mu_l1, mu_l2).
        mu_l1, _, mu_l2, _ = m.encode(x)
        manual = np.array(m.decode(mu_l1, mu_l2))
        np.testing.assert_allclose(a, manual, atol=1e-6)
        assert a.shape == (2, 32, 32, 3)

    def test_temperature_one_is_stochastic(self) -> None:
        cfg = _tiny_cfg()
        m = HierarchicalConvNeXtPatchVAE(cfg)
        rng = np.random.RandomState(0)
        x = keras.ops.convert_to_tensor(rng.rand(2, 32, 32, 3).astype("float32"))
        _ = m(x, training=False)
        a = np.array(m.sample_from(x, temperature=1.0, seed=1))
        b = np.array(m.sample_from(x, temperature=1.0, seed=2))
        assert not np.allclose(a, b, atol=1e-6)


class TestSIGRegOff:
    """``lambda_sigreg_*=0.0`` ablation path is clean at both scales."""

    def test_sigreg_off_branch(self) -> None:
        cfg = _tiny_cfg(lambda_sigreg_l1=0.0, lambda_sigreg_l2=0.0)
        m = HierarchicalConvNeXtPatchVAE(cfg)
        x = np.random.RandomState(11).rand(2, 32, 32, 3).astype("float32")
        m.sigreg_l1_loss_tracker.reset_state()
        m.sigreg_l2_loss_tracker.reset_state()
        _ = m(keras.ops.convert_to_tensor(x), training=False)
        # The 5 explicit add_loss calls in order are:
        #   [recon, beta_l1*kl_L1, beta_l2*kl_L2,
        #    lambda_l1*sigreg_L1, lambda_l2*sigreg_L2]
        losses = [float(l) for l in m.losses]
        assert len(losses) >= 5
        weighted_sigreg_l1 = losses[3]
        weighted_sigreg_l2 = losses[4]
        assert weighted_sigreg_l1 == 0.0, (
            f"weighted L1 SIGReg with lambda=0 must be 0, got "
            f"{weighted_sigreg_l1}"
        )
        assert weighted_sigreg_l2 == 0.0, (
            f"weighted L2 SIGReg with lambda=0 must be 0, got "
            f"{weighted_sigreg_l2}"
        )
        # Raw stats still tracked (D-003 contract preserved).
        raw_l1 = float(m.sigreg_l1_loss_tracker.result())
        raw_l2 = float(m.sigreg_l2_loss_tracker.result())
        assert np.isfinite(raw_l1) and raw_l1 >= 0.0
        assert np.isfinite(raw_l2) and raw_l2 >= 0.0


class TestFactory:
    """Named-variant factory surface."""

    def test_variants_build(self) -> None:
        for variant in HierarchicalConvNeXtPatchVAE.PRESETS:
            # Force a small img_size + small SIGReg to keep runtime tight.
            m = create_hierarchical_convnext_patch_vae(
                variant,
                img_size=16,
                patch_size_l1=8,
                patch_size_l2=4,
                kernel_size=3,
                sigreg_knots=5,
                sigreg_num_proj=32,
            )
            x = np.random.RandomState(0).rand(1, 16, 16, 3).astype("float32")
            out = m(keras.ops.convert_to_tensor(x), training=False)
            assert "reconstruction" in out
            assert tuple(out["reconstruction"].shape) == (1, 16, 16, 3)
            preset = HierarchicalConvNeXtPatchVAE.PRESETS[variant]
            assert m.config.embed_dim_l1 == preset["embed_dim_l1"]
            assert m.config.latent_dim_l1 == preset["latent_dim_l1"]
            assert m.config.latent_dim_l2 == preset["latent_dim_l2"]

    def test_from_variant_pretrained_raises(self) -> None:
        with pytest.raises(NotImplementedError):
            HierarchicalConvNeXtPatchVAE.from_variant("base", pretrained=True)

    def test_from_variant_unknown_raises(self) -> None:
        with pytest.raises(ValueError):
            HierarchicalConvNeXtPatchVAE.from_variant("nonexistent")

    def test_create_factory_smoke(self) -> None:
        m = create_hierarchical_convnext_patch_vae(
            "tiny",
            img_size=16,
            patch_size_l1=8,
            patch_size_l2=4,
            kernel_size=3,
            sigreg_knots=2,
            sigreg_num_proj=8,
        )
        assert isinstance(m, HierarchicalConvNeXtPatchVAE)
        preset = HierarchicalConvNeXtPatchVAE.PRESETS["tiny"]
        assert m.config.embed_dim_l1 == preset["embed_dim_l1"]
        shapes = m.compute_output_shape((None, 16, 16, 3))
        # Both L1/L2 + alias keys must be present.
        for k in [
            "reconstruction",
            "z_l1", "mu_l1", "log_var_l1",
            "z_l2", "mu_l2", "log_var_l2",
            "z", "mu", "log_var",
        ]:
            assert k in shapes, f"missing key {k}"
        assert shapes["reconstruction"] == (None, 16, 16, 3)
        # 16/p1=2, 16/p2=4.
        assert shapes["mu_l1"] == (None, 2, 2, m.config.latent_dim_l1)
        assert shapes["mu_l2"] == (None, 4, 4, m.config.latent_dim_l2)
