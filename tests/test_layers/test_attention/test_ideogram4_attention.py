"""
Test suite for Ideogram4Attention.

Covers: forward shape, the load-bearing block-diagonal segment-mask invariance
(segment-0 outputs are unaffected by perturbing segment-1 value inputs), QK-norm
presence + stability, finite gradients, ctor validation, and a full ``.keras``
serialization round-trip.
"""

import os
import tempfile

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.layers.attention.ideogram4_attention import Ideogram4Attention
from dl_techniques.layers.norms.rms_norm import RMSNorm
from dl_techniques.layers.embedding.multi_axis_rope import Ideogram4MRoPE


class TestIdeogram4Attention:
    """Test suite for the Ideogram4Attention layer."""

    HIDDEN = 64
    NUM_HEADS = 4
    HEAD_DIM = HIDDEN // NUM_HEADS  # 16

    @pytest.fixture
    def layer(self):
        return Ideogram4Attention(hidden_size=self.HIDDEN, num_heads=self.NUM_HEADS)

    @pytest.fixture
    def sample(self):
        """A (B=2, L=4, hidden) input with cos/sin and single-segment ids."""
        keras.utils.set_random_seed(42)
        b, l = 2, 4
        x = keras.random.normal((b, l, self.HIDDEN))
        cos = keras.ops.ones((b, l, self.HEAD_DIM))
        sin = keras.ops.zeros((b, l, self.HEAD_DIM))
        seg = keras.ops.zeros((b, l), dtype="int32")
        return x, seg, cos, sin

    # ------------------------------------------------------------------
    # Initialization / validation
    # ------------------------------------------------------------------

    def test_initialization(self, layer):
        assert layer.hidden_size == self.HIDDEN
        assert layer.num_heads == self.NUM_HEADS
        assert layer.head_dim == self.HEAD_DIM
        assert layer.eps == 1e-5
        assert layer.qkv is not None
        assert layer.o is not None

    def test_ctor_raises_on_indivisible(self):
        with pytest.raises(ValueError):
            Ideogram4Attention(hidden_size=65, num_heads=4)

    def test_ctor_raises_on_bad_num_heads(self):
        with pytest.raises(ValueError):
            Ideogram4Attention(hidden_size=64, num_heads=0)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def test_forward_shape(self, layer, sample):
        x, seg, cos, sin = sample
        y = layer(x, seg, cos, sin)
        assert tuple(y.shape) == (2, 4, self.HIDDEN)
        assert np.all(np.isfinite(keras.ops.convert_to_numpy(y)))

    def test_compute_output_shape(self, layer):
        assert layer.compute_output_shape((2, 4, self.HIDDEN)) == (2, 4, self.HIDDEN)

    # ------------------------------------------------------------------
    # QK-norm
    # ------------------------------------------------------------------

    def test_qk_norm_are_rmsnorm(self, layer):
        assert isinstance(layer.norm_q, RMSNorm)
        assert isinstance(layer.norm_k, RMSNorm)
        assert layer.norm_q.axis == -1
        assert layer.norm_k.axis == -1

    def test_qk_norm_scale_shape(self, layer, sample):
        x, seg, cos, sin = sample
        _ = layer(x, seg, cos, sin)  # trigger build
        assert tuple(layer.norm_q.scale.shape) == (self.HEAD_DIM,)
        assert tuple(layer.norm_k.scale.shape) == (self.HEAD_DIM,)

    def test_qk_norm_stabilizes_large_input(self, layer, sample):
        """Scaling the input by a large factor must not blow up the output
        (the RMS QK-norm makes attention logits scale-invariant in q/k)."""
        x, seg, cos, sin = sample
        y_small = keras.ops.convert_to_numpy(layer(x, seg, cos, sin))
        y_big = keras.ops.convert_to_numpy(layer(x * 100.0, seg, cos, sin))
        assert np.all(np.isfinite(y_big))
        # Output magnitude should remain comparable (v is linear in x, so the
        # output scales ~linearly, but must not explode beyond that scale).
        assert np.max(np.abs(y_big)) < 100.0 * np.max(np.abs(y_small)) * 10.0

    # ------------------------------------------------------------------
    # Block-diagonal segment mask (LOAD-BEARING)
    # ------------------------------------------------------------------

    def test_cross_segment_mask_invariance(self):
        """A segment-0 output token must be INVARIANT to changes in the inputs
        producing segment-1 value tokens. This proves the additive block-diagonal
        mask zeroes cross-segment attention."""
        keras.utils.set_random_seed(7)
        b, l = 1, 4
        layer = Ideogram4Attention(hidden_size=self.HIDDEN, num_heads=self.NUM_HEADS)

        seg = keras.ops.convert_to_tensor([[0, 0, 1, 1]], dtype="int32")
        cos = keras.ops.ones((b, l, self.HEAD_DIM))
        sin = keras.ops.zeros((b, l, self.HEAD_DIM))

        x1 = keras.random.normal((b, l, self.HIDDEN))
        y1 = keras.ops.convert_to_numpy(layer(x1, seg, cos, sin))

        # Perturb ONLY the segment-1 positions (indices 2, 3) of the input.
        x1_np = keras.ops.convert_to_numpy(x1).copy()
        x1_np[:, 2:, :] += keras.ops.convert_to_numpy(
            keras.random.normal((b, 2, self.HIDDEN)) * 5.0
        )
        x2 = keras.ops.convert_to_tensor(x1_np)
        y2 = keras.ops.convert_to_numpy(layer(x2, seg, cos, sin))

        # Segment-0 outputs (positions 0, 1) must be unchanged.
        np.testing.assert_allclose(y1[:, :2, :], y2[:, :2, :], atol=1e-5)
        # Sanity: segment-1 outputs DID change (otherwise the test is vacuous).
        assert not np.allclose(y1[:, 2:, :], y2[:, 2:, :], atol=1e-5)

    def test_mask_with_real_mrope(self):
        """Mask invariance still holds when cos/sin come from a real mRoPE."""
        keras.utils.set_random_seed(11)
        b, l = 1, 4
        layer = Ideogram4Attention(hidden_size=self.HIDDEN, num_heads=self.NUM_HEADS)
        mrope = Ideogram4MRoPE(
            head_dim=self.HEAD_DIM, rope_theta=10000.0, mrope_section=(2, 2, 2)
        )
        pos = keras.ops.convert_to_tensor(
            [[[0, 0, 0], [1, 0, 0], [0, 1, 1], [1, 1, 1]]], dtype="int32"
        )
        cos, sin = mrope(pos)
        seg = keras.ops.convert_to_tensor([[0, 0, 1, 1]], dtype="int32")

        x1 = keras.random.normal((b, l, self.HIDDEN))
        y1 = keras.ops.convert_to_numpy(layer(x1, seg, cos, sin))
        x1_np = keras.ops.convert_to_numpy(x1).copy()
        x1_np[:, 2:, :] += 5.0
        y2 = keras.ops.convert_to_numpy(layer(keras.ops.convert_to_tensor(x1_np), seg, cos, sin))
        np.testing.assert_allclose(y1[:, :2, :], y2[:, :2, :], atol=1e-5)

    # ------------------------------------------------------------------
    # Gradients
    # ------------------------------------------------------------------

    def test_gradients_finite(self, layer, sample):
        x, seg, cos, sin = sample
        x_var = tf.Variable(keras.ops.convert_to_numpy(x))
        with tf.GradientTape() as tape:
            y = layer(x_var, seg, cos, sin)
            loss = keras.ops.mean(keras.ops.square(y))
        grads = tape.gradient(loss, [x_var] + list(layer.trainable_variables))
        assert all(g is not None for g in grads)
        for g in grads:
            assert np.all(np.isfinite(keras.ops.convert_to_numpy(g)))

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def test_get_config_round_trip(self):
        layer = Ideogram4Attention(hidden_size=128, num_heads=8, eps=1e-4)
        cfg = layer.get_config()
        rebuilt = Ideogram4Attention.from_config(cfg)
        assert rebuilt.hidden_size == 128
        assert rebuilt.num_heads == 8
        assert rebuilt.eps == 1e-4

    def test_keras_serialization_round_trip(self, sample):
        x, seg, cos, sin = sample
        b, l = 2, 4

        x_in = keras.Input(shape=(l, self.HIDDEN), name="x")
        seg_in = keras.Input(shape=(l,), dtype="int32", name="seg")
        cos_in = keras.Input(shape=(l, self.HEAD_DIM), name="cos")
        sin_in = keras.Input(shape=(l, self.HEAD_DIM), name="sin")
        out = Ideogram4Attention(hidden_size=self.HIDDEN, num_heads=self.NUM_HEADS)(
            x_in, seg_in, cos_in, sin_in
        )
        model = keras.Model([x_in, seg_in, cos_in, sin_in], out)

        inputs = {
            "x": keras.ops.convert_to_numpy(x),
            "seg": keras.ops.convert_to_numpy(seg),
            "cos": keras.ops.convert_to_numpy(cos),
            "sin": keras.ops.convert_to_numpy(sin),
        }
        y_before = model.predict(inputs, verbose=0)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "attn.keras")
            model.save(path)
            reloaded = keras.models.load_model(path)
            y_after = reloaded.predict(inputs, verbose=0)

        np.testing.assert_allclose(y_before, y_after, atol=1e-6)
