"""Tests for V2 task heads (cls + seg)."""

from __future__ import annotations

import numpy as np
import pytest
import keras
from keras import ops

from dl_techniques.models.convnext_patch_vae_v2.heads import (
    AttentionPoolClassifierHead,
    SegmentationHead,
)


def _rand_pre_bottleneck(b=2, hp=4, wp=4, e=32, seed=0):
    rng = np.random.default_rng(seed)
    arr = rng.standard_normal(size=(b, hp, wp, e)).astype("float32")
    return ops.convert_to_tensor(arr)


# ---------------------------------------------------------------------------
# AttentionPoolClassifierHead
# ---------------------------------------------------------------------------


class TestAttentionPoolClassifierHead:

    def test_forward_shape(self):
        head = AttentionPoolClassifierHead(embed_dim=32, num_classes=10, num_heads=4)
        x = _rand_pre_bottleneck()
        logits = head(x, training=False)
        assert tuple(logits.shape) == (2, 10)

    def test_invalid_num_classes(self):
        with pytest.raises(ValueError):
            AttentionPoolClassifierHead(embed_dim=32, num_classes=1, num_heads=4)

    def test_num_heads_must_divide_embed_dim(self):
        with pytest.raises(ValueError, match="divide"):
            AttentionPoolClassifierHead(embed_dim=33, num_classes=10, num_heads=4)

    def test_get_config_keys(self):
        head = AttentionPoolClassifierHead(
            embed_dim=32, num_classes=10, num_heads=4, mlp_expansion=2,
            dropout_rate=0.1,
        )
        cfg = head.get_config()
        assert cfg["embed_dim"] == 32
        assert cfg["num_classes"] == 10
        assert cfg["num_heads"] == 4
        assert cfg["mlp_expansion"] == 2
        assert cfg["dropout_rate"] == pytest.approx(0.1)

    def test_resolution_agnostic(self):
        """Head works at arbitrary spatial size."""
        head = AttentionPoolClassifierHead(embed_dim=32, num_classes=5, num_heads=4)
        x_small = _rand_pre_bottleneck(hp=4, wp=4)
        x_big = _rand_pre_bottleneck(hp=8, wp=8, seed=1)
        out_small = head(x_small, training=False)
        out_big = head(x_big, training=False)
        assert tuple(out_small.shape) == (2, 5)
        assert tuple(out_big.shape) == (2, 5)

    def test_save_load_roundtrip(self, tmp_path):
        inp = keras.layers.Input(shape=(4, 4, 32))
        head = AttentionPoolClassifierHead(embed_dim=32, num_classes=10, num_heads=4)
        out = head(inp)
        model = keras.Model(inp, out)
        x = _rand_pre_bottleneck(seed=42)
        ref = model(x, training=False)
        path = tmp_path / "cls_head.keras"
        model.save(path)
        reloaded = keras.models.load_model(path)
        new = reloaded(x, training=False)
        np.testing.assert_allclose(
            np.array(ref), np.array(new), atol=1e-4
        )

    def test_gradient_flow(self):
        """Loss has gradients w.r.t. head trainable weights."""
        import tensorflow as tf
        head = AttentionPoolClassifierHead(embed_dim=32, num_classes=5, num_heads=4)
        x = _rand_pre_bottleneck()
        y = np.array([0, 1], dtype="int32")
        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        with tf.GradientTape() as tape:
            logits = head(x, training=True)
            loss = loss_fn(y, logits)
        grads = tape.gradient(loss, head.trainable_variables)
        assert len(grads) > 0
        assert all(g is not None for g in grads)


# ---------------------------------------------------------------------------
# SegmentationHead
# ---------------------------------------------------------------------------


class TestSegmentationHead:

    def test_forward_shape(self):
        head = SegmentationHead(embed_dim=32, num_classes=21, patch_size=4)
        x = _rand_pre_bottleneck(b=2, hp=8, wp=8, e=32)
        out = head(x, training=False)
        # Hp*patch_size = 32, Wp*patch_size = 32.
        assert tuple(out.shape) == (2, 32, 32, 21)

    def test_invalid_num_classes(self):
        with pytest.raises(ValueError):
            SegmentationHead(embed_dim=32, num_classes=1, patch_size=4)

    def test_invalid_patch_size(self):
        with pytest.raises(ValueError):
            SegmentationHead(embed_dim=32, num_classes=10, patch_size=0)

    def test_resolution_agnostic(self):
        head = SegmentationHead(embed_dim=32, num_classes=10, patch_size=4)
        x_small = _rand_pre_bottleneck(hp=4, wp=4)
        x_big = _rand_pre_bottleneck(hp=8, wp=8, seed=1)
        out_small = head(x_small, training=False)
        out_big = head(x_big, training=False)
        assert tuple(out_small.shape) == (2, 16, 16, 10)
        assert tuple(out_big.shape) == (2, 32, 32, 10)

    def test_save_load_roundtrip(self, tmp_path):
        inp = keras.layers.Input(shape=(4, 4, 32))
        head = SegmentationHead(embed_dim=32, num_classes=10, patch_size=4)
        out = head(inp)
        model = keras.Model(inp, out)
        x = _rand_pre_bottleneck(seed=42)
        ref = model(x, training=False)
        path = tmp_path / "seg_head.keras"
        model.save(path)
        reloaded = keras.models.load_model(path)
        new = reloaded(x, training=False)
        np.testing.assert_allclose(
            np.array(ref), np.array(new), atol=1e-4
        )

    def test_gradient_flow(self):
        import tensorflow as tf
        head = SegmentationHead(embed_dim=32, num_classes=5, patch_size=4)
        x = _rand_pre_bottleneck(b=2, hp=4, wp=4)
        # Synthetic per-pixel labels in [0, 5).
        y = np.random.RandomState(0).randint(0, 5, size=(2, 16, 16)).astype("int32")
        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        with tf.GradientTape() as tape:
            logits = head(x, training=True)
            loss = loss_fn(y, logits)
        grads = tape.gradient(loss, head.trainable_variables)
        assert len(grads) > 0
        assert all(g is not None for g in grads)
