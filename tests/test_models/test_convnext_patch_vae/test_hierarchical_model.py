"""Test suite for the hierarchical (2-level) ConvNeXt patch VAE.

Mirrors ``test_convnext_patch_vae.py`` structure for the pool-derived
two-level variant ``HierarchicalConvNeXtPatchVAE`` introduced in
``plans/plan_2026-06-08_e3917bd5/`` (fine ``z1`` / coarse ``z2``, learned
top-down conditional prior ``p(z1|z2)``, free-bits gate on the coarse KL).

Hardest-first order (plan §5 / Success Criteria SC1-SC6):

1. Config round-trip + invariant violations (TestConfig).
2. Resolution-agnostic forward — all 10 keys at res 32 & 48 (SC1) and
   the two-level per-patch KL ratio gate (SC2) (TestForward).
3. Conditional-prior step-0 N(0,I) identity (SC3) (TestConditionalPriorIdentity).
4. Free-bits floor active on the coarse KL (SC4) (TestFreeBits).
5. ``.keras`` save/load round-trip on deterministic ``encode`` mu (SC5)
   (TestSaveLoad).
6. ``fit`` reports a finite, positive loss (SC6) (TestFit).
7. ``sample`` / ``sample_from`` / ``decode`` API (TestSampleAPI).
8. Named-variant factory surface (TestFactory).
"""

from __future__ import annotations

import os

import keras
import numpy as np
import pytest

from dl_techniques.layers.convnext_v2_block import ConvNextV2Block
from dl_techniques.layers.sampling import Sampling
from dl_techniques.regularizers.sigreg import SIGRegLayer
from dl_techniques.models.convnext_patch_vae.config import (
    HierarchicalConvNeXtPatchVAEConfig,
)
from dl_techniques.models.convnext_patch_vae.decoder import ConvNeXtPatchDecoder
from dl_techniques.models.convnext_patch_vae.encoder import ConvNeXtPatchEncoder
from dl_techniques.models.convnext_patch_vae.model_hierarchical import (
    HierarchicalConvNeXtPatchVAE,
    _CoarseLatentHead,
    _L2ConditionalPrior,
    create_hierarchical_convnext_patch_vae,
)

# All registered classes a reloaded hierarchical model may reference. They
# are ``@register_keras_serializable`` (resolved automatically on import),
# but we pass them explicitly to keras.models.load_model so the test does
# not silently depend on import-time registration order.
_CUSTOM_OBJECTS = {
    "HierarchicalConvNeXtPatchVAE": HierarchicalConvNeXtPatchVAE,
    "_L2ConditionalPrior": _L2ConditionalPrior,
    "_CoarseLatentHead": _CoarseLatentHead,
    "ConvNeXtPatchEncoder": ConvNeXtPatchEncoder,
    "ConvNeXtPatchDecoder": ConvNeXtPatchDecoder,
    "ConvNextV2Block": ConvNextV2Block,
    "SIGRegLayer": SIGRegLayer,
    "Sampling": Sampling,
}


def _tiny_hier_cfg(**overrides) -> HierarchicalConvNeXtPatchVAEConfig:
    """Small pool-derived hierarchical config — fast unit-test scale.

    img_size=32, patch_size=4 -> fine grid 8x8 (even), coarse 4x4. With
    ``sigreg_knots=5`` the coarse num_patches (16) >= knots, so the
    SIGReg advisory warning stays silent.
    """
    base = dict(
        img_size=32,
        patch_size=4,
        embed_dim=16,
        encoder_depth=1,
        decoder_depth=1,
        kernel_size=3,
        latent_dim=8,
        coarse_latent_dim=8,
        prior_depth=1,
        recon_loss_type="mse",
        sigreg_knots=5,
        sigreg_num_proj=16,
    )
    base.update(overrides)
    return HierarchicalConvNeXtPatchVAEConfig(**base)


class TestConfig:
    """``HierarchicalConvNeXtPatchVAEConfig`` round-trip + invariants."""

    def test_config_roundtrip(self) -> None:
        cfg = _tiny_hier_cfg()
        d = cfg.to_dict()
        cfg2 = HierarchicalConvNeXtPatchVAEConfig.from_dict(d)
        assert cfg.to_dict() == cfg2.to_dict()
        # Derived properties survive the round-trip.
        assert cfg.patches_per_side == 8
        assert cfg.coarse_patches_per_side == 4
        assert cfg.num_patches == 64
        assert cfg.input_image_shape == (32, 32, 3)
        # Sentinel prior_embed_dim (0) resolves to embed_dim.
        assert cfg.effective_prior_embed_dim == 16

    def test_effective_prior_embed_dim_override(self) -> None:
        cfg = _tiny_hier_cfg(prior_embed_dim=24)
        assert cfg.effective_prior_embed_dim == 24

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"img_size": 36, "patch_size": 4},   # Hp=9 odd -> pool/up mismatch
            {"img_size": 33, "patch_size": 4},   # not divisible by patch_size
            {"latent_dim": 0},                   # zero-dim fine latent rejected
            {"coarse_latent_dim": 0},            # zero-dim coarse latent rejected
            {"recon_loss_type": "bad"},          # unknown family
            {"beta_kl_l1": -1.0},                # negative fine KL weight
            {"beta_kl_l2": -1.0},                # negative coarse KL weight
            {"lambda_sigreg_l2": -0.5},          # negative coarse SIGReg weight
            {"free_bits": -1.0},                 # negative free-bits floor
            {"prior_depth": 0},                  # prior must have >= 1 block
            {"sigreg_knots": 1},                 # below SIGReg floor
            {"dropout_rate": 1.5},               # out of [0, 1]
        ],
    )
    def test_invariant_violations_reject(self, kwargs: dict) -> None:
        with pytest.raises(ValueError):
            _tiny_hier_cfg(**kwargs)


class TestForward:
    """SC1 — all 10 output keys/shapes at res 32 & 48; SC2 — KL ratio gate."""

    def test_output_keys_and_shapes(self) -> None:
        # --- Resolution 32: fine 8x8, coarse 4x4. ---
        cfg32 = _tiny_hier_cfg()
        m32 = HierarchicalConvNeXtPatchVAE(cfg32)
        x32 = np.random.RandomState(0).rand(2, 32, 32, 3).astype("float32")
        out32 = m32(keras.ops.convert_to_tensor(x32), training=False)
        # All 10 documented keys present.
        expected_keys = {
            "reconstruction", "z", "mu", "log_var",
            "z1", "z2", "mu1", "mu2", "log_var1", "log_var2",
        }
        assert set(out32.keys()) == expected_keys
        # Recon matches input.
        assert tuple(out32["reconstruction"].shape) == (2, 32, 32, 3)
        # Fine grid 8x8 x D1=8.
        for k in ("z1", "mu1", "log_var1", "z", "mu", "log_var"):
            assert tuple(out32[k].shape) == (2, 8, 8, 8), k
        # Coarse grid 4x4 x D2=8.
        for k in ("z2", "mu2", "log_var2"):
            assert tuple(out32[k].shape) == (2, 4, 4, 8), k

        # --- Resolution 48: fine 12x12, coarse 6x6 (second model). ---
        cfg48 = _tiny_hier_cfg(img_size=48)
        m48 = HierarchicalConvNeXtPatchVAE(cfg48)
        x48 = np.random.RandomState(1).rand(2, 48, 48, 3).astype("float32")
        out48 = m48(keras.ops.convert_to_tensor(x48), training=False)
        assert set(out48.keys()) == expected_keys
        assert tuple(out48["reconstruction"].shape) == (2, 48, 48, 3)
        for k in ("z1", "mu1", "log_var1"):
            assert tuple(out48[k].shape) == (2, 12, 12, 8), k
        for k in ("z2", "mu2", "log_var2"):
            assert tuple(out48[k].shape) == (2, 6, 6, 8), k

    def test_kl_ratio_resolution_invariant(self) -> None:
        # img_size is fixed per config, so build two same-arch models at
        # res 32 and 48 and read each one's per-patch-mean two-level KL
        # (kl_l1 + kl_l2) from the trackers after a forward call.
        cfg32 = _tiny_hier_cfg()
        cfg48 = _tiny_hier_cfg(img_size=48)
        m32 = HierarchicalConvNeXtPatchVAE(cfg32)
        m48 = HierarchicalConvNeXtPatchVAE(cfg48)
        rng = np.random.RandomState(7)
        x32 = rng.rand(4, 32, 32, 3).astype("float32")
        x48 = rng.rand(4, 48, 48, 3).astype("float32")

        def _kl_total(m, x):
            m.kl_l1_loss_tracker.reset_state()
            m.kl_l2_loss_tracker.reset_state()
            _ = m(keras.ops.convert_to_tensor(x), training=False)
            return (
                float(m.kl_l1_loss_tracker.result())
                + float(m.kl_l2_loss_tracker.result())
            )

        kl_32 = _kl_total(m32, x32)
        kl_48 = _kl_total(m48, x48)
        # Per-patch averaging: a larger grid must not scale the KL up.
        ratio = max(kl_32, kl_48) / max(min(kl_32, kl_48), 1e-9)
        assert ratio < 2.0, (
            f"two-level per-patch KL not resolution-invariant: "
            f"32->{kl_32}, 48->{kl_48}, ratio={ratio}"
        )


class TestConditionalPriorIdentity:
    """SC3 — conditional-prior step-0 N(0,I) identity (bit-exact)."""

    def test_step0_identity(self) -> None:
        cfg = _tiny_hier_cfg()
        m = HierarchicalConvNeXtPatchVAE(cfg)
        rng = np.random.RandomState(0)
        x = keras.ops.convert_to_tensor(
            rng.rand(2, 32, 32, 3).astype("float32")
        )
        # Build the graph so all sub-layers (and zero-init heads) exist.
        _ = m(x, training=False)

        # Tap the encoder, coarse head, sample z2, and the prior.
        delta_mu, lv1, h = m.encoder(x, training=False, return_features=True)
        mu2, lv2 = m.coarse_head(h, training=False)
        z2 = m.sampling([mu2, lv2], training=False)
        mu_p, lv_p = m.prior(z2, training=False)

        mu_p_np = keras.ops.convert_to_numpy(mu_p)
        lv_p_np = keras.ops.convert_to_numpy(lv_p)
        # Zero-init prior heads => p(z1|z2) = N(0, I) for ANY z2.
        assert np.max(np.abs(mu_p_np)) < 1e-5, (
            f"mu_p not ~0 at step 0: max|mu_p|={np.max(np.abs(mu_p_np))}"
        )
        assert np.max(np.abs(lv_p_np)) < 1e-5, (
            f"log_var_p not ~0 at step 0: max|lv_p|={np.max(np.abs(lv_p_np))}"
        )

        # VDVAE delta: mu1 = mu_p + delta_mu.
        mu1 = mu_p + delta_mu

        # Model's conditional KL.
        kl_cond = float(
            keras.ops.convert_to_numpy(
                m._compute_kl_l2_conditional(mu1, lv1, mu_p, lv_p)
            )
        )
        # Standard N(0,I) KL computed inline, same reduction (per-patch sum
        # over D1, then grid mean).
        mu1_np = keras.ops.convert_to_numpy(mu1).astype(np.float64)
        lv1_np = np.clip(
            keras.ops.convert_to_numpy(lv1).astype(np.float64), -10.0, 10.0
        )
        kl_per_patch = -0.5 * np.sum(
            1.0 + lv1_np - np.square(mu1_np) - np.exp(lv1_np), axis=-1
        )
        kl_standard = float(np.mean(kl_per_patch))
        assert abs(kl_cond - kl_standard) < 1e-5, (
            f"step-0 conditional KL != N(0,I) KL: cond={kl_cond}, "
            f"standard={kl_standard}, diff={abs(kl_cond - kl_standard)}"
        )


class TestFreeBits:
    """SC4 — free-bits floor active on the coarse KL when raw < floor."""

    def test_floor_active(self) -> None:
        # At init the coarse posterior is near N(0,I) (lv2 zero-init, mu2
        # small), so the raw coarse per-patch KL is tiny. A large free_bits
        # forces the gated coarse KL up to the floor.
        m_hi = HierarchicalConvNeXtPatchVAE(_tiny_hier_cfg(free_bits=100.0))
        m_lo = HierarchicalConvNeXtPatchVAE(_tiny_hier_cfg(free_bits=0.0))
        rng = np.random.RandomState(0)
        x = keras.ops.convert_to_tensor(
            rng.rand(2, 32, 32, 3).astype("float32")
        )

        def _kl_l2(m):
            m.kl_l2_loss_tracker.reset_state()
            _ = m(x, training=False)
            return float(m.kl_l2_loss_tracker.result())

        kl_hi = _kl_l2(m_hi)
        kl_lo = _kl_l2(m_lo)
        # Gated coarse KL is clamped to the 100-nat floor.
        assert kl_hi >= 100.0 - 1e-3, (
            f"free_bits=100 coarse KL not floored: {kl_hi}"
        )
        # The un-floored (free_bits=0) raw coarse KL is >= 0 and below the
        # floored value.
        assert kl_lo >= 0.0
        assert kl_lo < kl_hi, (
            f"free_bits=0 coarse KL ({kl_lo}) must be < floored "
            f"coarse KL ({kl_hi})"
        )


class TestSaveLoad:
    """SC5 — full ``.keras`` save/load round-trip on deterministic encode mu."""

    def test_encode_mu_roundtrip(self, tmp_path) -> None:
        cfg = _tiny_hier_cfg()
        m = HierarchicalConvNeXtPatchVAE(cfg)
        rng = np.random.RandomState(42)
        x = rng.rand(2, 32, 32, 3).astype("float32")
        xt = keras.ops.convert_to_tensor(x)
        # Build the full graph (all sub-layer weights) before save.
        _ = m(xt, training=False)
        mu_before, _ = m.encode(xt)
        mu_before_np = keras.ops.convert_to_numpy(mu_before)

        path = os.path.join(str(tmp_path), "h_cpvae.keras")
        m.save(path)
        loaded = keras.models.load_model(path, custom_objects=_CUSTOM_OBJECTS)

        mu_after, _ = loaded.encode(xt)
        mu_after_np = keras.ops.convert_to_numpy(mu_after)
        np.testing.assert_allclose(mu_after_np, mu_before_np, atol=1e-4)
        # Config invariants survive the round-trip.
        assert loaded.config.to_dict() == cfg.to_dict()


class TestFit:
    """SC6 — ``history.history['loss']`` finite and > 0 after one epoch."""

    def test_one_epoch_loss_finite(self) -> None:
        cfg = _tiny_hier_cfg()
        m = HierarchicalConvNeXtPatchVAE(cfg)
        m.compile(optimizer=keras.optimizers.Adam(1e-3), loss=None)
        x = np.random.RandomState(3).rand(4, 32, 32, 3).astype("float32")
        hist = m.fit(x, x, epochs=1, batch_size=2, verbose=0)
        loss = hist.history["loss"][0]
        assert np.isfinite(loss), f"loss must be finite, got {loss}"
        assert loss > 0.0, f"loss must be > 0, got {loss}"


class TestSampleAPI:
    """``sample`` / ``sample_from`` / ``decode`` resolution-agnostic API."""

    def test_sample_default_grid(self) -> None:
        cfg = _tiny_hier_cfg()
        m = HierarchicalConvNeXtPatchVAE(cfg)
        s = m.sample(num_samples=2, seed=0)
        # Default grid = patches_per_side (8) -> 8 * patch_size(4) = 32.
        assert tuple(s.shape) == (2, 32, 32, 3)

    def test_sample_explicit_grid(self) -> None:
        cfg = _tiny_hier_cfg()
        m = HierarchicalConvNeXtPatchVAE(cfg)
        s = m.sample(num_samples=2, hp=8, wp=8, seed=0)
        assert tuple(s.shape) == (2, 32, 32, 3)

    def test_sample_from_temperature_zero_deterministic(self) -> None:
        cfg = _tiny_hier_cfg()
        m = HierarchicalConvNeXtPatchVAE(cfg)
        rng = np.random.RandomState(0)
        x = keras.ops.convert_to_tensor(
            rng.rand(2, 32, 32, 3).astype("float32")
        )
        _ = m(x, training=False)  # build
        a = np.array(m.sample_from(x, temperature=0.0, seed=1))
        b = np.array(m.sample_from(x, temperature=0.0, seed=2))
        # t=0 ignores the noise -> deterministic.
        np.testing.assert_allclose(a, b, atol=1e-6)
        assert a.shape == (2, 32, 32, 3)

    def test_sample_from_temperature_one_runs(self) -> None:
        cfg = _tiny_hier_cfg()
        m = HierarchicalConvNeXtPatchVAE(cfg)
        rng = np.random.RandomState(0)
        x = keras.ops.convert_to_tensor(
            rng.rand(2, 32, 32, 3).astype("float32")
        )
        _ = m(x, training=False)
        out = np.array(m.sample_from(x, temperature=1.0, seed=1))
        assert out.shape == (2, 32, 32, 3)
        assert np.all(np.isfinite(out))

    def test_decode_encode_shape(self) -> None:
        cfg = _tiny_hier_cfg()
        m = HierarchicalConvNeXtPatchVAE(cfg)
        rng = np.random.RandomState(0)
        x = keras.ops.convert_to_tensor(
            rng.rand(2, 32, 32, 3).astype("float32")
        )
        _ = m(x, training=False)
        mu1, _ = m.encode(x)
        recon = m.decode(mu1)
        assert tuple(recon.shape) == (2, 32, 32, 3)


class TestFactory:
    """Named-variant factory surface (PRESETS, from_variant, create_*)."""

    @pytest.mark.parametrize(
        "variant,embed_dim",
        [("tiny", 64), ("base", 128), ("large", 192)],
    )
    def test_variants_build(self, variant: str, embed_dim: int) -> None:
        # Force a small img_size to keep runtime tight; preset values still
        # drive embed_dim / depth / latent_dim.
        m = create_hierarchical_convnext_patch_vae(
            variant,
            img_size=16,
            patch_size=4,
            sigreg_knots=2,
            sigreg_num_proj=16,
        )
        x = np.random.RandomState(0).rand(1, 16, 16, 3).astype("float32")
        out = m(keras.ops.convert_to_tensor(x), training=False)
        assert "reconstruction" in out
        assert tuple(out["reconstruction"].shape) == (1, 16, 16, 3)
        assert m.config.embed_dim == embed_dim

    def test_from_variant_pretrained_raises(self) -> None:
        with pytest.raises(NotImplementedError):
            HierarchicalConvNeXtPatchVAE.from_variant("base", pretrained=True)

    def test_from_variant_unknown_raises(self) -> None:
        with pytest.raises(ValueError):
            HierarchicalConvNeXtPatchVAE.from_variant("nonexistent")

    def test_compute_output_shape_keys(self) -> None:
        cfg = _tiny_hier_cfg()
        m = HierarchicalConvNeXtPatchVAE(cfg)
        shapes = m.compute_output_shape((None, 32, 32, 3))
        assert set(shapes.keys()) == {
            "reconstruction", "z", "mu", "log_var",
            "z1", "z2", "mu1", "mu2", "log_var1", "log_var2",
        }
        assert shapes["reconstruction"] == (None, 32, 32, 3)
        # Fine grid 8x8 x D1=8; coarse 4x4 x D2=8.
        assert shapes["mu1"] == (None, 8, 8, 8)
        assert shapes["mu2"] == (None, 4, 4, 8)
