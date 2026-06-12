"""Tests for Ideogram4TransformerBlock and Ideogram4FinalLayer."""

import os
import tempfile

import keras
import numpy as np
import pytest

from dl_techniques.layers.embedding.multi_axis_rope import Ideogram4MRoPE
from dl_techniques.layers.transformers.ideogram4_block import (
    Ideogram4TransformerBlock,
    Ideogram4FinalLayer,
)

# ---------------------------------------------------------------------
# Test config: small dims that respect hidden_size % num_heads == 0 and
# head_dim divisible by the mRoPE sections.
# ---------------------------------------------------------------------
HIDDEN = 64
NUM_HEADS = 4
HEAD_DIM = HIDDEN // NUM_HEADS  # 16
INTERMEDIATE = 128
ADALN_DIM = 32
OUT_CHANNELS = 8
BATCH = 2
LENGTH = 6
MROPE_SECTION = (3, 2, 2)  # over head_dim/2 = 8


def _rope_tables(batch=BATCH, length=LENGTH):
    """Build real cos/sin tables via Ideogram4MRoPE of correct head_dim."""
    rope = Ideogram4MRoPE(
        head_dim=HEAD_DIM, rope_theta=10_000.0, mrope_section=MROPE_SECTION
    )
    position_ids = keras.ops.cast(
        np.tile(np.arange(length)[None, :, None], (batch, 1, 3)), "int32"
    )
    cos, sin = rope(position_ids)
    return cos, sin


def _segment_ids(batch=BATCH, length=LENGTH):
    return keras.ops.zeros((batch, length), dtype="int32")


def _inputs(adaln_len=1, batch=BATCH, length=LENGTH):
    x = keras.random.normal((batch, length, HIDDEN), seed=0)
    seg = _segment_ids(batch, length)
    cos, sin = _rope_tables(batch, length)
    adaln = keras.random.normal((batch, adaln_len, ADALN_DIM), seed=1)
    return x, seg, cos, sin, adaln


def _make_block():
    return Ideogram4TransformerBlock(
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_heads=NUM_HEADS,
        adaln_dim=ADALN_DIM,
    )


def _make_final():
    return Ideogram4FinalLayer(
        hidden_size=HIDDEN, out_channels=OUT_CHANNELS, adaln_dim=ADALN_DIM
    )


# =====================================================================
# Block tests
# =====================================================================
class TestIdeogram4TransformerBlock:
    def test_swiglu_hidden_dim_exact(self):
        """The reused SwiGLUFFN must have hidden_dim == intermediate_size."""
        block = _make_block()
        assert block.feed_forward.hidden_dim == INTERMEDIATE
        assert block.feed_forward.use_bias is False

    def test_output_shape_adaln_rank3_per_sample(self):
        block = _make_block()
        x, seg, cos, sin, adaln = _inputs(adaln_len=1)
        out = block(x, seg, cos, sin, adaln)
        assert tuple(out.shape) == (BATCH, LENGTH, HIDDEN)

    def test_output_shape_adaln_per_token(self):
        block = _make_block()
        x, seg, cos, sin, adaln = _inputs(adaln_len=LENGTH)
        out = block(x, seg, cos, sin, adaln)
        assert tuple(out.shape) == (BATCH, LENGTH, HIDDEN)

    def test_identity_at_zero_modulation(self):
        """LOAD-BEARING: zero adaln kernel+bias => scale=1, gate=0 => identity.

        With adaln_modulation outputting all zeros, scale_msa=scale_mlp=1 and
        gate_msa=gate_mlp=tanh(0)=0, so both gated residual branches vanish and
        the block returns x unchanged.
        """
        block = _make_block()
        x, seg, cos, sin, adaln = _inputs(adaln_len=1)
        # Force build, then zero the adaln_modulation weights.
        _ = block(x, seg, cos, sin, adaln)
        kernel, bias = block.adaln_modulation.get_weights()
        block.adaln_modulation.set_weights(
            [np.zeros_like(kernel), np.zeros_like(bias)]
        )
        out = block(x, seg, cos, sin, adaln)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(out),
            keras.ops.convert_to_numpy(x),
            atol=1e-6,
        )

    def test_modulation_numpy_reference(self):
        """LOAD-BEARING: verify scale=1+x and gate=tanh(x) numerically.

        Set the adaln_modulation kernel to zero and the bias to a known vector
        b so mod == b everywhere. Then chunk b into 4 and reproduce the full
        block forward in numpy (scale=1+chunk, gate=tanh(chunk)), comparing to
        the layer output at atol 1e-6.
        """
        block = _make_block()
        x, seg, cos, sin, adaln = _inputs(adaln_len=1)
        _ = block(x, seg, cos, sin, adaln)

        rng = np.random.default_rng(0)
        kernel, bias = block.adaln_modulation.get_weights()
        # Small bias so tanh is well inside its linear-ish region but nonzero.
        b = rng.uniform(-0.5, 0.5, size=bias.shape).astype(bias.dtype)
        block.adaln_modulation.set_weights([np.zeros_like(kernel), b])

        out = keras.ops.convert_to_numpy(block(x, seg, cos, sin, adaln))

        # numpy reproduction of the modulation, using the layer's own
        # deterministic sublayers for attn/norm/ffn.
        chunks = np.split(b, 4)  # each (hidden,)
        scale_msa = 1.0 + chunks[0]
        gate_msa = np.tanh(chunks[1])
        scale_mlp = 1.0 + chunks[2]
        gate_mlp = np.tanh(chunks[3])

        x_np = keras.ops.convert_to_numpy(x)

        def to_np(t):
            return keras.ops.convert_to_numpy(t)

        # attention branch
        attn_norm1 = to_np(block.attention_norm1(x))
        attn_in = attn_norm1 * scale_msa
        attn_out = to_np(
            block.attention(
                keras.ops.convert_to_tensor(attn_in), seg, cos, sin
            )
        )
        attn_norm2 = to_np(
            block.attention_norm2(keras.ops.convert_to_tensor(attn_out))
        )
        x1 = x_np + gate_msa * attn_norm2

        # ffn branch
        ffn_norm1 = to_np(block.ffn_norm1(keras.ops.convert_to_tensor(x1)))
        ffn_in = ffn_norm1 * scale_mlp
        ffn_out = to_np(block.feed_forward(keras.ops.convert_to_tensor(ffn_in)))
        ffn_norm2 = to_np(block.ffn_norm2(keras.ops.convert_to_tensor(ffn_out)))
        x2 = x1 + gate_mlp * ffn_norm2

        np.testing.assert_allclose(out, x2, atol=1e-6)

    def test_gate_bounded_by_tanh(self):
        """With a huge modulation magnitude, gates saturate (|tanh| <= 1)."""
        block = _make_block()
        x, seg, cos, sin, adaln = _inputs(adaln_len=1)
        _ = block(x, seg, cos, sin, adaln)
        kernel, bias = block.adaln_modulation.get_weights()
        big = np.full_like(bias, 100.0)
        block.adaln_modulation.set_weights([np.zeros_like(kernel), big])
        # gate path = tanh(100) ~ 1; the residual update is finite (bounded).
        out = keras.ops.convert_to_numpy(block(x, seg, cos, sin, adaln))
        assert np.all(np.isfinite(out))

    def test_gradients_finite(self):
        import tensorflow as tf

        block = _make_block()
        x, seg, cos, sin, adaln = _inputs(adaln_len=1)
        x_var = tf.Variable(keras.ops.convert_to_numpy(x))
        with tf.GradientTape() as tape:
            out = block(x_var, seg, cos, sin, adaln)
            loss = keras.ops.mean(keras.ops.square(out))
        grads = tape.gradient(loss, [x_var] + list(block.trainable_variables))
        for g in grads:
            assert g is not None
            assert np.all(np.isfinite(keras.ops.convert_to_numpy(g)))

    def test_ctor_validation_num_heads(self):
        with pytest.raises(ValueError):
            Ideogram4TransformerBlock(
                hidden_size=65,
                intermediate_size=INTERMEDIATE,
                num_heads=NUM_HEADS,
                adaln_dim=ADALN_DIM,
            )

    def test_ctor_validation_intermediate_too_small(self):
        with pytest.raises(ValueError):
            Ideogram4TransformerBlock(
                hidden_size=64,
                intermediate_size=1,  # < int(64*2/3) = 42
                num_heads=NUM_HEADS,
                adaln_dim=ADALN_DIM,
            )

    def test_serialization_round_trip(self):
        block = _make_block()
        x, seg, cos, sin, adaln = _inputs(adaln_len=1)

        x_in = keras.Input(batch_shape=(BATCH, LENGTH, HIDDEN))
        seg_in = keras.Input(batch_shape=(BATCH, LENGTH), dtype="int32")
        cos_in = keras.Input(batch_shape=(BATCH, LENGTH, HEAD_DIM))
        sin_in = keras.Input(batch_shape=(BATCH, LENGTH, HEAD_DIM))
        adaln_in = keras.Input(batch_shape=(BATCH, 1, ADALN_DIM))
        out = block(x_in, seg_in, cos_in, sin_in, adaln_in)
        model = keras.Model([x_in, seg_in, cos_in, sin_in, adaln_in], out)

        ref = model([x, seg, cos, sin, adaln])

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "block.keras")
            model.save(path)
            reloaded = keras.models.load_model(path)
            got = reloaded([x, seg, cos, sin, adaln])

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(ref),
            keras.ops.convert_to_numpy(got),
            atol=1e-6,
        )


# =====================================================================
# Final layer tests
# =====================================================================
class TestIdeogram4FinalLayer:
    def test_output_shape(self):
        final = _make_final()
        x = keras.random.normal((BATCH, LENGTH, HIDDEN), seed=2)
        c = keras.random.normal((BATCH, 1, ADALN_DIM), seed=3)
        out = final(x, c)
        assert tuple(out.shape) == (BATCH, LENGTH, OUT_CHANNELS)

    def test_output_shape_per_token_cond(self):
        final = _make_final()
        x = keras.random.normal((BATCH, LENGTH, HIDDEN), seed=2)
        c = keras.random.normal((BATCH, LENGTH, ADALN_DIM), seed=3)
        out = final(x, c)
        assert tuple(out.shape) == (BATCH, LENGTH, OUT_CHANNELS)

    def test_scale_one_at_zero_adaln(self):
        """Zero adaln kernel+bias => scale=1 => output == linear(layernorm(x))."""
        final = _make_final()
        x = keras.random.normal((BATCH, LENGTH, HIDDEN), seed=4)
        c = keras.random.normal((BATCH, 1, ADALN_DIM), seed=5)
        _ = final(x, c)  # build
        kernel, bias = final.adaln_modulation.get_weights()
        final.adaln_modulation.set_weights(
            [np.zeros_like(kernel), np.zeros_like(bias)]
        )
        out = keras.ops.convert_to_numpy(final(x, c))

        normed = final.norm_final(x)
        expected = keras.ops.convert_to_numpy(final.linear(normed))
        np.testing.assert_allclose(out, expected, atol=1e-6)

    def test_gradients_finite(self):
        import tensorflow as tf

        final = _make_final()
        x = keras.random.normal((BATCH, LENGTH, HIDDEN), seed=6)
        c = keras.random.normal((BATCH, 1, ADALN_DIM), seed=7)
        x_var = tf.Variable(keras.ops.convert_to_numpy(x))
        with tf.GradientTape() as tape:
            out = final(x_var, c)
            loss = keras.ops.mean(keras.ops.square(out))
        grads = tape.gradient(loss, [x_var] + list(final.trainable_variables))
        for g in grads:
            assert g is not None
            assert np.all(np.isfinite(keras.ops.convert_to_numpy(g)))

    def test_ctor_validation(self):
        with pytest.raises(ValueError):
            Ideogram4FinalLayer(
                hidden_size=-1, out_channels=OUT_CHANNELS, adaln_dim=ADALN_DIM
            )

    def test_serialization_round_trip(self):
        final = _make_final()
        x = keras.random.normal((BATCH, LENGTH, HIDDEN), seed=8)
        c = keras.random.normal((BATCH, 1, ADALN_DIM), seed=9)

        x_in = keras.Input(batch_shape=(BATCH, LENGTH, HIDDEN))
        c_in = keras.Input(batch_shape=(BATCH, 1, ADALN_DIM))
        out = final(x_in, c_in)
        model = keras.Model([x_in, c_in], out)

        ref = model([x, c])
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "final.keras")
            model.save(path)
            reloaded = keras.models.load_model(path)
            got = reloaded([x, c])

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(ref),
            keras.ops.convert_to_numpy(got),
            atol=1e-6,
        )
