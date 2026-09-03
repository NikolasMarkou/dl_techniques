"""Tests for ``LeVJEPABlock`` (forward shape, RoPE, gradient flow, round-trip)."""

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.models.vision.levjepa.blocks import LeVJEPABlock

from ..gradient_flow_oracle import assert_gradients_reach_every_trainable_weight


class _BlockWrapper(keras.layers.Layer):
    """Adapts ``LeVJEPABlock.call``'s extra kwargs to the oracle's
    ``model(inputs, training=...)`` contract."""

    def __init__(self, block: LeVJEPABlock, num_frames: int, height_patches: int, width_patches: int, **kwargs):
        super().__init__(**kwargs)
        self.block = block
        self._num_frames = num_frames
        self._height_patches = height_patches
        self._width_patches = width_patches

    def call(self, inputs, training=None):
        return self.block(
            inputs,
            num_frames=self._num_frames,
            height_patches=self._height_patches,
            width_patches=self._width_patches,
            training=training,
        )


class TestLeVJEPABlockForward:
    def test_forward_shape_no_rope(self):
        block = LeVJEPABlock(dim=32, num_heads=4, use_rope=False)
        x = keras.random.normal((2, 10, 32))
        out = block(x)
        assert out.shape == (2, 10, 32)

    def test_forward_shape_with_rope(self):
        # head_dim = 32 // 4 = 8; band = 2 * ((8 // 3) // 2) = 2 > 0.
        block = LeVJEPABlock(dim=32, num_heads=4, use_rope=True, num_prefix_tokens=1)
        num_frames, h_patches, w_patches = 2, 3, 3
        num_tokens = 1 + num_frames * h_patches * w_patches
        x = keras.random.normal((2, num_tokens, 32))
        out = block(
            x, num_frames=num_frames, height_patches=h_patches, width_patches=w_patches
        )
        assert out.shape == (2, num_tokens, 32)

    def test_use_rope_requires_grid_dims(self):
        block = LeVJEPABlock(dim=32, num_heads=4, use_rope=True)
        x = keras.random.normal((2, 10, 32))
        with pytest.raises(ValueError, match="requires height_patches and width_patches"):
            block(x)

    def test_attn_mask_is_honoured(self):
        # A mask that forbids everything except self-attention (identity
        # diagonal) must not crash and must still respect finite outputs.
        block = LeVJEPABlock(dim=16, num_heads=2, use_rope=False)
        x = keras.random.normal((1, 5, 16))
        keep = np.eye(5, dtype=bool)[None, None, :, :]  # (1, 1, 5, 5)
        out = block(x, attn_mask=keras.ops.convert_to_tensor(keep))
        assert out.shape == (1, 5, 16)
        assert np.all(np.isfinite(keras.ops.convert_to_numpy(out)))


class TestLeVJEPABlockGradientFlow:
    @pytest.mark.parametrize("use_rope", [False, True])
    def test_gradients_reach_every_trainable_weight_after_one_step(self, use_rope):
        num_frames, h_patches, w_patches = 2, 3, 3
        num_tokens = 1 + num_frames * h_patches * w_patches
        block = LeVJEPABlock(dim=32, num_heads=4, use_rope=use_rope, num_prefix_tokens=1)
        wrapper = _BlockWrapper(block, num_frames, h_patches, w_patches)

        rng = np.random.default_rng(0)
        x = rng.standard_normal((2, num_tokens, 32)).astype("float32")

        wrapper(x, training=True)  # build

        optimizer = keras.optimizers.Adam(1e-3)
        with tf.GradientTape() as tape:
            out = wrapper(x, training=True)
            loss = keras.ops.mean(keras.ops.square(out))
        grads = tape.gradient(loss, wrapper.trainable_weights)
        optimizer.apply_gradients(zip(grads, wrapper.trainable_weights))

        report = assert_gradients_reach_every_trainable_weight(wrapper, x)
        assert len(report) == len(wrapper.trainable_weights)


class TestLeVJEPABlockSerialization:
    def test_round_trip_config(self):
        block = LeVJEPABlock(
            dim=24, num_heads=3, use_rope=True, mlp_ratio=2.0,
            layer_id=2, dropout_rate=0.1, attention_dropout_rate=0.1,
        )
        x = keras.random.normal((1, 10, 24))
        block(x, num_frames=1, height_patches=3, width_patches=3)

        clone = LeVJEPABlock.from_config(block.get_config())
        clone.build(x.shape)
        assert len(clone.get_weights()) == len(block.get_weights())
        for a, b in zip(clone.get_weights(), block.get_weights()):
            assert a.shape == b.shape

        clone.set_weights(block.get_weights())

        out1 = block(x, num_frames=1, height_patches=3, width_patches=3, training=False)
        out2 = clone(x, num_frames=1, height_patches=3, width_patches=3, training=False)
        max_delta = float(
            np.max(np.abs(keras.ops.convert_to_numpy(out1) - keras.ops.convert_to_numpy(out2)))
        )
        assert max_delta == 0.0
