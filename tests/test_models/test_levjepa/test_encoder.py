"""Tests for ``LeVJEPAEncoder`` (image/video paths, masking, RoPE, round-trip)."""

import keras
import numpy as np
import pytest

from dl_techniques.models.vision.levjepa.encoder import LeVJEPAEncoder


def _small_kwargs(**overrides):
    cfg = dict(
        input_shape=(32, 32, 3),
        patch_size=16,
        embed_dim=16,
        depth=2,
        num_heads=2,
    )
    cfg.update(overrides)
    return cfg


class TestLeVJEPAEncoderForward:
    def test_image_path_forward_shape(self):
        enc = LeVJEPAEncoder(**_small_kwargs(num_frames=1))
        x = keras.random.normal((2, 32, 32, 3))
        out = enc(x)
        # (32/16)^2 = 4 patches + 1 CLS
        assert out.shape == (2, 5, 16)

    def test_video_path_forward_shape(self):
        enc = LeVJEPAEncoder(**_small_kwargs(num_frames=4, tubelet_size=2))
        x = keras.random.normal((2, 4, 32, 32, 3))
        out = enc(x)
        # t_patches=2, h_patches=w_patches=2 -> 2*2*2=8 patches + 1 CLS
        assert out.shape == (2, 9, 16)

    @pytest.mark.parametrize("attn_mode", ["full", "block_causal"])
    def test_attn_mode_both_produce_valid_output(self, attn_mode):
        enc = LeVJEPAEncoder(
            **_small_kwargs(num_frames=4, tubelet_size=2, attn_mode=attn_mode)
        )
        x = keras.random.normal((2, 4, 32, 32, 3))
        out = enc(x)
        assert out.shape == (2, 9, 16)
        assert np.all(np.isfinite(keras.ops.convert_to_numpy(out)))

    @pytest.mark.parametrize("use_rope", [False, True])
    def test_use_rope_both_work(self, use_rope):
        enc = LeVJEPAEncoder(
            **_small_kwargs(num_frames=4, tubelet_size=2, use_rope=use_rope)
        )
        x = keras.random.normal((2, 4, 32, 32, 3))
        out = enc(x)
        assert out.shape == (2, 9, 16)
        assert np.all(np.isfinite(keras.ops.convert_to_numpy(out)))

    def test_token_dropping_active_only_when_training(self):
        enc = LeVJEPAEncoder(**_small_kwargs(num_frames=1, token_dropout_rate=0.5))
        x = keras.random.normal((2, 32, 32, 3))

        out_infer = enc(x, training=False)
        # 4 patches + 1 CLS, no dropping at inference.
        assert out_infer.shape == (2, 5, 16)

        out_train = enc(x, training=True)
        # dropout_rate=0.5 over 4 patches -> keep_len = round(4*0.5) = 2, + 1 CLS.
        assert out_train.shape == (2, 3, 16)

    def test_token_drop_rate_zero_is_identity_length(self):
        enc = LeVJEPAEncoder(**_small_kwargs(num_frames=1, token_dropout_rate=0.0))
        x = keras.random.normal((2, 32, 32, 3))
        out_train = enc(x, training=True)
        assert out_train.shape == (2, 5, 16)


class TestLeVJEPAEncoderMutualExclusion:
    def test_use_rope_true_builds_no_pos_embed(self):
        # DECISION plan-2026-09-03T113223-2a714a91/D-013 resolution: use_rope is the
        # ONLY toggle, and use_rope=True must leave pos_embed unbuilt (None).
        enc = LeVJEPAEncoder(**_small_kwargs(num_frames=1, use_rope=True))
        x = keras.random.normal((2, 32, 32, 3))
        enc(x)
        assert enc.built
        assert enc.pos_embed is None

    def test_use_rope_false_builds_a_pos_embed_weight(self):
        enc = LeVJEPAEncoder(**_small_kwargs(num_frames=1, use_rope=False))
        x = keras.random.normal((2, 32, 32, 3))
        enc(x)
        assert enc.built
        assert enc.pos_embed is not None
        assert enc.pos_embed.trainable is False


class TestLeVJEPAEncoderSerialization:
    def test_round_trip_config(self):
        enc = LeVJEPAEncoder(**_small_kwargs(num_frames=1))
        x = keras.random.normal((1, 32, 32, 3))
        enc(x)

        clone = LeVJEPAEncoder.from_config(enc.get_config())
        clone(x)

        assert len(clone.get_weights()) == len(enc.get_weights())
        for a, b in zip(clone.get_weights(), enc.get_weights()):
            assert a.shape == b.shape

        clone.set_weights(enc.get_weights())

        out1 = enc(x, training=False)
        out2 = clone(x, training=False)
        max_delta = float(
            np.max(np.abs(keras.ops.convert_to_numpy(out1) - keras.ops.convert_to_numpy(out2)))
        )
        assert max_delta == 0.0

    def test_invalid_attn_mode_raises(self):
        with pytest.raises(ValueError, match="attn_mode"):
            LeVJEPAEncoder(**_small_kwargs(num_frames=1, attn_mode="bogus"))
