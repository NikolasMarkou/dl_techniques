"""Tests for ConvNeXtPatchEncoderV2."""

from __future__ import annotations

import numpy as np
import pytest
import keras
from keras import ops

from dl_techniques.models.convnext_patch_vae_v2.encoder import (
    ConvNeXtPatchEncoderV2,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_kwargs():
    return dict(
        patch_size=4,
        embed_dim=32,
        depth=2,
        kernel_size=7,
        latent_dim=8,
    )


def _rand_image(b=2, h=16, w=16, c=3, seed=0):
    rng = np.random.default_rng(seed)
    arr = rng.uniform(0.0, 1.0, size=(b, h, w, c)).astype("float32")
    return ops.convert_to_tensor(arr)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestForward:

    def test_two_tuple_default(self, tiny_kwargs):
        enc = ConvNeXtPatchEncoderV2(**tiny_kwargs)
        x = _rand_image(2, 16, 16)
        mu, log_var = enc(x, training=False)
        assert tuple(mu.shape) == (2, 4, 4, 8)
        assert tuple(log_var.shape) == (2, 4, 4, 8)

    def test_log_var_zero_init(self, tiny_kwargs):
        """log_var head must be zero-init (V1 D-003 convention preserved)."""
        enc = ConvNeXtPatchEncoderV2(**tiny_kwargs)
        x = _rand_image(2, 16, 16)
        _, log_var = enc(x, training=False)
        np.testing.assert_allclose(
            np.array(log_var), np.zeros((2, 4, 4, 8)), atol=1e-6
        )

    def test_four_tuple_with_pre_bottleneck(self, tiny_kwargs):
        enc = ConvNeXtPatchEncoderV2(**tiny_kwargs)
        x = _rand_image(2, 16, 16)
        mu, log_var, pre, mask = enc(x, training=False, output_pre_bottleneck=True)
        assert tuple(mu.shape) == (2, 4, 4, 8)
        assert tuple(pre.shape) == (2, 4, 4, 32)
        assert mask is None  # no mask when training=False

    def test_resolution_agnostic(self, tiny_kwargs):
        """Same encoder instance, two resolutions, both work."""
        enc = ConvNeXtPatchEncoderV2(**tiny_kwargs)
        x_a = _rand_image(2, 16, 16)
        x_b = _rand_image(2, 32, 32, seed=1)
        mu_a, _ = enc(x_a, training=False)
        mu_b, _ = enc(x_b, training=False)
        assert tuple(mu_a.shape) == (2, 4, 4, 8)
        assert tuple(mu_b.shape) == (2, 8, 8, 8)

    def test_non_divisible_input_raises(self, tiny_kwargs):
        enc = ConvNeXtPatchEncoderV2(**tiny_kwargs)
        # 17 is not divisible by patch_size=4.
        with pytest.raises(ValueError, match="divisible"):
            _ = enc.build((None, 17, 16, 3))


class TestMAEMasking:

    def test_mask_token_not_created_when_ratio_zero(self, tiny_kwargs):
        enc = ConvNeXtPatchEncoderV2(**tiny_kwargs, mae_mask_ratio=0.0)
        x = _rand_image(2, 16, 16)
        _ = enc(x, training=True)
        assert enc.mask_token is None

    def test_mask_token_created_when_ratio_positive(self, tiny_kwargs):
        enc = ConvNeXtPatchEncoderV2(**tiny_kwargs, mae_mask_ratio=0.5)
        x = _rand_image(2, 16, 16)
        _ = enc(x, training=True)
        assert enc.mask_token is not None
        # (1, 1, 1, embed_dim).
        assert tuple(enc.mask_token.shape) == (1, 1, 1, 32)

    def test_mask_returned_in_4tuple_when_training(self, tiny_kwargs):
        enc = ConvNeXtPatchEncoderV2(
            **tiny_kwargs, mae_mask_ratio=0.5, mae_mask_seed=42
        )
        x = _rand_image(2, 16, 16)
        _, _, _, mask = enc(x, training=True, output_pre_bottleneck=True)
        assert mask is not None
        assert tuple(mask.shape) == (2, 4, 4, 1)
        # Exactly half of 16 = 8 patches masked per sample.
        per_sample = np.array(mask).reshape(2, -1).sum(axis=1)
        np.testing.assert_array_equal(per_sample, np.array([8, 8]))

    def test_mask_is_none_when_not_training(self, tiny_kwargs):
        enc = ConvNeXtPatchEncoderV2(**tiny_kwargs, mae_mask_ratio=0.5)
        x = _rand_image(2, 16, 16)
        _, _, _, mask = enc(x, training=False, output_pre_bottleneck=True)
        assert mask is None

    def test_ratio_zero_path_is_v1_equivalent_with_pre_bottleneck(self, tiny_kwargs):
        """Encoder built with mae_mask_ratio=0 should produce identical
        outputs in training=True and training=False (no random mask)."""
        keras.utils.set_random_seed(7)
        enc = ConvNeXtPatchEncoderV2(**tiny_kwargs, mae_mask_ratio=0.0)
        x = _rand_image(2, 16, 16)
        # Drop dropout/spatial_dropout effects: tiny_kwargs sets them to 0,
        # so training=True/False outputs should match.
        mu_eval, _ = enc(x, training=False)
        mu_train, _ = enc(x, training=True)
        np.testing.assert_allclose(np.array(mu_eval), np.array(mu_train), atol=1e-6)


class TestConfigRoundtrip:

    def test_get_config_keys(self, tiny_kwargs):
        enc = ConvNeXtPatchEncoderV2(**tiny_kwargs, mae_mask_ratio=0.5, mae_mask_seed=11)
        cfg = enc.get_config()
        assert cfg["patch_size"] == 4
        assert cfg["embed_dim"] == 32
        assert cfg["depth"] == 2
        assert cfg["latent_dim"] == 8
        assert cfg["mae_mask_ratio"] == 0.5
        assert cfg["mae_mask_seed"] == 11

    def test_from_config(self, tiny_kwargs):
        enc = ConvNeXtPatchEncoderV2(**tiny_kwargs, mae_mask_ratio=0.5)
        cfg = enc.get_config()
        enc2 = ConvNeXtPatchEncoderV2.from_config(cfg)
        assert enc2.patch_size == enc.patch_size
        assert enc2.embed_dim == enc.embed_dim
        assert enc2.mae_mask_ratio == enc.mae_mask_ratio


class TestSaveLoadRoundtrip:

    def test_keras_save_load_roundtrip(self, tiny_kwargs, tmp_path):
        """Wrap in keras.Model + .keras save / load round-trip on mu (atol=1e-4)."""
        inp = keras.layers.Input(shape=(16, 16, 3))
        enc = ConvNeXtPatchEncoderV2(**tiny_kwargs, name="enc")
        mu, log_var = enc(inp)
        model = keras.Model(inp, [mu, log_var])
        # warm up — required for save round-trip on un-called subclass nodes
        x = _rand_image(2, 16, 16, seed=99)
        ref_mu, _ = model(x)

        path = tmp_path / "enc_v2.keras"
        model.save(path)
        reloaded = keras.models.load_model(path)
        new_mu, _ = reloaded(x)
        max_delta = float(np.max(np.abs(np.array(ref_mu) - np.array(new_mu))))
        assert max_delta < 1e-4
