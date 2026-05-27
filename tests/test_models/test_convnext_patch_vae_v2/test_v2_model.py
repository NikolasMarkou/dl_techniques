"""Tests for ConvNeXtPatchVAEV2."""

from __future__ import annotations

import numpy as np
import pytest
import keras
from keras import ops

from dl_techniques.models.convnext_patch_vae_v2.config import (
    ConvNeXtPatchVAEV2Config,
)
from dl_techniques.models.convnext_patch_vae_v2.model import (
    ConvNeXtPatchVAEV2,
    create_convnext_patch_vae_v2,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _vae_only_cfg(**overrides) -> ConvNeXtPatchVAEV2Config:
    """Tiny VAE-only config — V1-equivalent (all heads off, MAE=0, LPIPS=0)."""
    base = dict(
        img_size=16,
        img_channels=3,
        patch_size=4,
        embed_dim=32,
        encoder_depth=2,
        decoder_depth=2,
        kernel_size=7,
        latent_dim=8,
        beta_kl=0.5,
        lambda_sigreg=0.1,
        sigreg_knots=4,
        sigreg_num_proj=32,
        recon_loss_type="mse",
    )
    base.update(overrides)
    return ConvNeXtPatchVAEV2Config(**base)


def _rand_image(b=2, h=16, w=16, c=3, seed=0, low=0.0, high=1.0):
    rng = np.random.default_rng(seed)
    return ops.convert_to_tensor(
        rng.uniform(low, high, size=(b, h, w, c)).astype("float32")
    )


# ---------------------------------------------------------------------------
# V1 compatibility
# ---------------------------------------------------------------------------


class TestV1Compatibility:

    def test_v1_compat_output_dict(self):
        """All flags off → output dict has only the V1 keys + shapes."""
        cfg = _vae_only_cfg()
        model = ConvNeXtPatchVAEV2(config=cfg)
        x = _rand_image()
        out = model(x, training=False)
        assert set(out.keys()) == {"reconstruction", "z", "mu", "log_var"}
        assert tuple(out["reconstruction"].shape) == (2, 16, 16, 3)
        assert tuple(out["z"].shape) == (2, 4, 4, 8)
        assert tuple(out["mu"].shape) == (2, 4, 4, 8)

    def test_one_fit_step_produces_nonzero_loss(self):
        cfg = _vae_only_cfg()
        model = ConvNeXtPatchVAEV2(config=cfg)
        model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=None, jit_compile=False)
        x = np.array(_rand_image(b=4))
        hist = model.fit(x, x, epochs=1, batch_size=2, verbose=0, steps_per_epoch=2)
        assert "loss" in hist.history
        assert hist.history["loss"][0] > 0.0
        assert np.isfinite(hist.history["loss"][0])


# ---------------------------------------------------------------------------
# MAE masking active
# ---------------------------------------------------------------------------


class TestMAEPath:

    def test_mae_active_recon_loss_finite(self):
        cfg = _vae_only_cfg(mae_mask_ratio=0.5, lambda_mae=1.0)
        model = ConvNeXtPatchVAEV2(config=cfg)
        model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=None, jit_compile=False)
        x = np.array(_rand_image(b=4))
        hist = model.fit(x, x, epochs=1, batch_size=2, verbose=0, steps_per_epoch=2)
        assert hist.history["loss"][0] > 0.0
        assert hist.history["mae_loss"][0] > 0.0
        assert np.isfinite(hist.history["mae_loss"][0])

    def test_mae_ratio_zero_keeps_mae_loss_zero(self):
        cfg = _vae_only_cfg(mae_mask_ratio=0.0)
        model = ConvNeXtPatchVAEV2(config=cfg)
        model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=None, jit_compile=False)
        x = np.array(_rand_image(b=4))
        hist = model.fit(x, x, epochs=1, batch_size=2, verbose=0, steps_per_epoch=2)
        # mae_loss tracker exists but stays at 0.
        assert hist.history["mae_loss"][0] == pytest.approx(0.0, abs=1e-7)


# ---------------------------------------------------------------------------
# Classification head
# ---------------------------------------------------------------------------


class TestClassificationHead:

    def test_logits_shape(self):
        cfg = _vae_only_cfg(use_classification_head=True, num_classes_cls=10)
        model = ConvNeXtPatchVAEV2(config=cfg)
        x = _rand_image()
        out = model(x, training=False)
        assert "logits_cls" in out
        assert tuple(out["logits_cls"].shape) == (2, 10)

    def test_fit_with_labels_via_dict_input(self):
        cfg = _vae_only_cfg(use_classification_head=True, num_classes_cls=5)
        model = ConvNeXtPatchVAEV2(config=cfg)
        model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=None, jit_compile=False)

        import tensorflow as tf

        x = np.array(_rand_image(b=8))
        y = np.random.RandomState(1).randint(0, 5, size=(8,)).astype("int32")

        ds = tf.data.Dataset.from_tensor_slices(
            ({"image": x, "label_cls": y}, x)
        ).batch(4).repeat()

        hist = model.fit(ds, epochs=1, steps_per_epoch=2, verbose=0)
        assert hist.history["loss"][0] > 0.0
        assert hist.history["cls_loss"][0] > 0.0
        assert np.isfinite(hist.history["cls_loss"][0])

    def test_logits_returned_when_no_label_present(self):
        """cls_loss stays 0 but logits are still produced."""
        cfg = _vae_only_cfg(use_classification_head=True, num_classes_cls=5)
        model = ConvNeXtPatchVAEV2(config=cfg)
        # Dict input WITHOUT label_cls.
        out = model({"image": _rand_image()}, training=False)
        assert "logits_cls" in out
        assert tuple(out["logits_cls"].shape) == (2, 5)


# ---------------------------------------------------------------------------
# Segmentation head
# ---------------------------------------------------------------------------


class TestSegmentationHead:

    def test_logits_shape(self):
        cfg = _vae_only_cfg(use_segmentation_head=True, num_classes_seg=4)
        model = ConvNeXtPatchVAEV2(config=cfg)
        x = _rand_image()
        out = model(x, training=False)
        assert "logits_seg" in out
        assert tuple(out["logits_seg"].shape) == (2, 16, 16, 4)

    def test_fit_with_seg_labels(self):
        cfg = _vae_only_cfg(use_segmentation_head=True, num_classes_seg=3)
        model = ConvNeXtPatchVAEV2(config=cfg)
        model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=None, jit_compile=False)

        import tensorflow as tf

        x = np.array(_rand_image(b=4))
        y = np.random.RandomState(2).randint(0, 3, size=(4, 16, 16)).astype("int32")

        ds = tf.data.Dataset.from_tensor_slices(
            ({"image": x, "label_seg": y}, x)
        ).batch(2).repeat()
        hist = model.fit(ds, epochs=1, steps_per_epoch=2, verbose=0)
        assert hist.history["seg_loss"][0] > 0.0
        assert np.isfinite(hist.history["seg_loss"][0])


# ---------------------------------------------------------------------------
# LPIPS active
# ---------------------------------------------------------------------------


class TestLPIPS:

    def test_lpips_active_loss_finite(self):
        # Skip if VGG weights unavailable.
        try:
            keras.applications.VGG16(include_top=False, weights="imagenet")
        except Exception as exc:  # pragma: no cover
            pytest.skip(f"VGG16 weights unavailable: {exc}")

        cfg = _vae_only_cfg(
            lambda_lpips=0.1,
            recon_loss_type="bce",
            lpips_input_range=(0.0, 1.0),
        )
        model = ConvNeXtPatchVAEV2(config=cfg)
        model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=None, jit_compile=False)
        x = np.array(_rand_image(b=4))
        hist = model.fit(x, x, epochs=1, batch_size=2, verbose=0, steps_per_epoch=2)
        assert hist.history["lpips_loss"][0] > 0.0
        assert np.isfinite(hist.history["lpips_loss"][0])


# ---------------------------------------------------------------------------
# All-on integration
# ---------------------------------------------------------------------------


class TestAllOn:

    def test_multi_task_fit_all_heads(self):
        try:
            keras.applications.VGG16(include_top=False, weights="imagenet")
        except Exception as exc:  # pragma: no cover
            pytest.skip(f"VGG16 weights unavailable: {exc}")

        cfg = _vae_only_cfg(
            recon_loss_type="bce",
            mae_mask_ratio=0.5,
            lambda_mae=1.0,
            lambda_lpips=0.1,
            use_classification_head=True,
            num_classes_cls=5,
            use_segmentation_head=True,
            num_classes_seg=3,
        )
        model = ConvNeXtPatchVAEV2(config=cfg)
        model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=None, jit_compile=False)

        import tensorflow as tf

        x = np.array(_rand_image(b=8))
        y_cls = np.random.RandomState(3).randint(0, 5, size=(8,)).astype("int32")
        y_seg = np.random.RandomState(4).randint(0, 3, size=(8, 16, 16)).astype("int32")

        ds = tf.data.Dataset.from_tensor_slices(
            ({"image": x, "label_cls": y_cls, "label_seg": y_seg}, x)
        ).batch(4).repeat()
        hist = model.fit(ds, epochs=1, steps_per_epoch=2, verbose=0)
        for k in ("loss", "recon_loss", "kl_loss", "sigreg_loss",
                  "mae_loss", "lpips_loss", "cls_loss", "seg_loss"):
            assert k in hist.history
            assert np.isfinite(hist.history[k][0])


# ---------------------------------------------------------------------------
# Resolution invariance
# ---------------------------------------------------------------------------


class TestResolutionInvariance:

    def test_train_at_16_forward_at_32(self):
        cfg = _vae_only_cfg()
        model = ConvNeXtPatchVAEV2(config=cfg)
        # Build at 16x16.
        _ = model(_rand_image(b=2, h=16, w=16), training=False)
        # Forward at 32x32 on the SAME instance.
        out_big = model(_rand_image(b=2, h=32, w=32, seed=1), training=False)
        assert tuple(out_big["reconstruction"].shape) == (2, 32, 32, 3)
        assert tuple(out_big["mu"].shape) == (2, 8, 8, 8)


# ---------------------------------------------------------------------------
# Save / load round-trip on mu
# ---------------------------------------------------------------------------


class TestSaveLoad:

    @pytest.mark.parametrize("variant_kwargs", [
        {},  # VAE-only
        {"mae_mask_ratio": 0.5},
        {"use_classification_head": True, "num_classes_cls": 5},
        {"use_segmentation_head": True, "num_classes_seg": 3},
    ])
    def test_save_load_mu_roundtrip(self, variant_kwargs, tmp_path):
        cfg = _vae_only_cfg(**variant_kwargs)
        model = ConvNeXtPatchVAEV2(config=cfg)
        x = _rand_image(b=2, seed=99)
        _ = model(x, training=False)
        ref_mu, _ = model.encode(x)

        path = tmp_path / "v2_model.keras"
        model.save(path)
        reloaded = keras.models.load_model(path)
        new_mu, _ = reloaded.encode(x)
        max_delta = float(np.max(np.abs(np.array(ref_mu) - np.array(new_mu))))
        assert max_delta < 1e-4

    def test_save_load_with_lpips(self, tmp_path):
        try:
            keras.applications.VGG16(include_top=False, weights="imagenet")
        except Exception as exc:  # pragma: no cover
            pytest.skip(f"VGG16 weights unavailable: {exc}")
        cfg = _vae_only_cfg(lambda_lpips=0.1, recon_loss_type="bce")
        model = ConvNeXtPatchVAEV2(config=cfg)
        x = _rand_image(b=2, seed=100)
        _ = model(x, training=False)
        ref_mu, _ = model.encode(x)
        path = tmp_path / "v2_lpips.keras"
        model.save(path)
        reloaded = keras.models.load_model(path)
        new_mu, _ = reloaded.encode(x)
        max_delta = float(np.max(np.abs(np.array(ref_mu) - np.array(new_mu))))
        assert max_delta < 1e-4


# ---------------------------------------------------------------------------
# Factory + variant
# ---------------------------------------------------------------------------


class TestFactoryAndVariants:

    def test_unknown_variant_raises(self):
        with pytest.raises(ValueError, match="Unknown variant"):
            create_convnext_patch_vae_v2(variant="not_a_variant")

    def test_pretrained_raises(self):
        with pytest.raises(NotImplementedError):
            create_convnext_patch_vae_v2(variant="tiny", pretrained=True)

    def test_tiny_variant_smoke(self):
        # Override img_size so the factory works for a smoke without huge model.
        model = create_convnext_patch_vae_v2(variant="tiny", img_size=16, patch_size=4)
        x = _rand_image(b=1)
        out = model(x, training=False)
        assert tuple(out["reconstruction"].shape) == (1, 16, 16, 3)

    def test_xl_preset_exists(self):
        # `xl` preset exists with expected dims.
        from dl_techniques.models.convnext_patch_vae_v2.config import PRESETS
        assert "xl" in PRESETS
        assert PRESETS["xl"]["embed_dim"] == 256
        assert PRESETS["xl"]["latent_dim"] == 64


# ---------------------------------------------------------------------------
# Sampling API parity with V1
# ---------------------------------------------------------------------------


class TestSamplingAPI:

    def test_sample_returns_image(self):
        cfg = _vae_only_cfg()
        model = ConvNeXtPatchVAEV2(config=cfg)
        # warmup
        _ = model(_rand_image(b=1), training=False)
        out = model.sample(num_samples=3, hp=4, wp=4)
        assert tuple(out.shape) == (3, 16, 16, 3)

    def test_sample_from(self):
        cfg = _vae_only_cfg()
        model = ConvNeXtPatchVAEV2(config=cfg)
        x = _rand_image(b=2, seed=7)
        out = model.sample_from(x, temperature=0.0)
        assert tuple(out.shape) == (2, 16, 16, 3)
