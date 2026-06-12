"""
Test suite for MMDiTJointAttention (SD3 dual-stream joint attention).

Covers: instantiation + ctor validation, dual-stream forward shapes,
``context_pre_only`` single-stream return, ``compute_output_shape`` pre/post
build, ``get_config``/``from_config`` round-trip, per-head QK-norm scale shape,
finite gradients, a full ``.keras`` save/load round-trip via a two-input
Functional model, and variable batch / variable N_txt.
"""

import os
import tempfile

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.layers.attention.mmdit_joint_attention import MMDiTJointAttention
from dl_techniques.layers.norms.rms_norm import RMSNorm


class TestMMDiTJointAttention:
    """Test suite for the MMDiTJointAttention layer."""

    DIM = 64
    NUM_HEADS = 4
    HEAD_DIM = DIM // NUM_HEADS  # 16
    N_IMG = 16
    N_TXT = 7
    BATCH = 2

    @pytest.fixture
    def layer(self):
        return MMDiTJointAttention(dim=self.DIM, num_heads=self.NUM_HEADS)

    @pytest.fixture
    def sample(self):
        """(B, N_img, dim) image stream + (B, N_txt, dim) text stream."""
        keras.utils.set_random_seed(42)
        img = keras.random.normal((self.BATCH, self.N_IMG, self.DIM))
        txt = keras.random.normal((self.BATCH, self.N_TXT, self.DIM))
        return img, txt

    # ------------------------------------------------------------------
    # Initialization / validation
    # ------------------------------------------------------------------

    def test_initialization(self, layer):
        assert layer.dim == self.DIM
        assert layer.num_heads == self.NUM_HEADS
        assert layer.head_dim == self.HEAD_DIM
        assert layer.qk_norm is True
        assert layer.use_bias is True
        assert layer.context_pre_only is False
        assert layer.eps == 1e-6
        for proj in (layer.to_q, layer.to_k, layer.to_v, layer.to_out,
                     layer.add_q_proj, layer.add_k_proj, layer.add_v_proj):
            assert proj is not None
        assert layer.to_add_out is not None

    def test_ctor_raises_on_indivisible(self):
        with pytest.raises(ValueError):
            MMDiTJointAttention(dim=65, num_heads=4)

    def test_ctor_raises_on_bad_num_heads(self):
        with pytest.raises(ValueError):
            MMDiTJointAttention(dim=64, num_heads=0)

    def test_ctor_raises_on_bad_eps(self):
        with pytest.raises(ValueError):
            MMDiTJointAttention(dim=64, num_heads=4, eps=0.0)

    def test_context_pre_only_drops_add_out(self):
        layer = MMDiTJointAttention(
            dim=self.DIM, num_heads=self.NUM_HEADS, context_pre_only=True
        )
        assert layer.to_add_out is None

    def test_qk_norm_disabled(self):
        layer = MMDiTJointAttention(
            dim=self.DIM, num_heads=self.NUM_HEADS, qk_norm=False
        )
        assert layer.norm_q is None
        assert layer.norm_added_k is None

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def test_forward_dual_stream_shapes(self, layer, sample):
        img, txt = sample
        img_out, txt_out = layer([img, txt])
        assert tuple(img_out.shape) == (self.BATCH, self.N_IMG, self.DIM)
        assert tuple(txt_out.shape) == (self.BATCH, self.N_TXT, self.DIM)
        assert np.all(np.isfinite(keras.ops.convert_to_numpy(img_out)))
        assert np.all(np.isfinite(keras.ops.convert_to_numpy(txt_out)))

    def test_forward_context_pre_only_single_stream(self, sample):
        img, txt = sample
        layer = MMDiTJointAttention(
            dim=self.DIM, num_heads=self.NUM_HEADS, context_pre_only=True
        )
        out = layer([img, txt])
        # Single tensor, not a list.
        assert not isinstance(out, (list, tuple))
        assert tuple(out.shape) == (self.BATCH, self.N_IMG, self.DIM)

    def test_forward_no_qk_norm(self, sample):
        img, txt = sample
        layer = MMDiTJointAttention(
            dim=self.DIM, num_heads=self.NUM_HEADS, qk_norm=False
        )
        img_out, txt_out = layer([img, txt])
        assert tuple(img_out.shape) == (self.BATCH, self.N_IMG, self.DIM)
        assert tuple(txt_out.shape) == (self.BATCH, self.N_TXT, self.DIM)

    def test_forward_no_bias(self, sample):
        img, txt = sample
        layer = MMDiTJointAttention(
            dim=self.DIM, num_heads=self.NUM_HEADS, use_bias=False
        )
        img_out, _ = layer([img, txt])
        assert tuple(img_out.shape) == (self.BATCH, self.N_IMG, self.DIM)

    # ------------------------------------------------------------------
    # compute_output_shape
    # ------------------------------------------------------------------

    def test_compute_output_shape_before_build(self):
        """Must work without ever building the layer."""
        layer = MMDiTJointAttention(dim=self.DIM, num_heads=self.NUM_HEADS)
        out = layer.compute_output_shape(
            [(self.BATCH, self.N_IMG, self.DIM), (self.BATCH, self.N_TXT, self.DIM)]
        )
        assert out == [
            (self.BATCH, self.N_IMG, self.DIM),
            (self.BATCH, self.N_TXT, self.DIM),
        ]

    def test_compute_output_shape_context_pre_only(self):
        layer = MMDiTJointAttention(
            dim=self.DIM, num_heads=self.NUM_HEADS, context_pre_only=True
        )
        out = layer.compute_output_shape(
            [(self.BATCH, self.N_IMG, self.DIM), (self.BATCH, self.N_TXT, self.DIM)]
        )
        assert out == (self.BATCH, self.N_IMG, self.DIM)

    def test_compute_output_shape_matches_actual(self, layer, sample):
        img, txt = sample
        img_out, txt_out = layer([img, txt])
        computed = layer.compute_output_shape(
            [tuple(img.shape), tuple(txt.shape)]
        )
        assert computed[0] == tuple(img_out.shape)
        assert computed[1] == tuple(txt_out.shape)

    # ------------------------------------------------------------------
    # QK-norm shape
    # ------------------------------------------------------------------

    def test_qk_norm_are_rmsnorm(self, layer):
        for n in (layer.norm_q, layer.norm_k, layer.norm_added_q, layer.norm_added_k):
            assert isinstance(n, RMSNorm)
            assert n.axis == -1

    def test_qk_norm_scale_shape(self, layer, sample):
        img, txt = sample
        _ = layer([img, txt])  # trigger build
        for n in (layer.norm_q, layer.norm_k, layer.norm_added_q, layer.norm_added_k):
            assert tuple(n.scale.shape) == (self.HEAD_DIM,)

    # ------------------------------------------------------------------
    # Gradients
    # ------------------------------------------------------------------

    def test_gradients_finite(self, layer, sample):
        img, txt = sample
        img_var = tf.Variable(keras.ops.convert_to_numpy(img))
        txt_var = tf.Variable(keras.ops.convert_to_numpy(txt))
        with tf.GradientTape() as tape:
            img_out, txt_out = layer([img_var, txt_var])
            loss = keras.ops.mean(keras.ops.square(img_out)) + keras.ops.mean(
                keras.ops.square(txt_out)
            )
        grads = tape.gradient(
            loss, [img_var, txt_var] + list(layer.trainable_variables)
        )
        assert all(g is not None for g in grads)
        for g in grads:
            assert np.all(np.isfinite(keras.ops.convert_to_numpy(g)))

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def test_get_config_contains_all_args(self, layer):
        cfg = layer.get_config()
        for key in ("dim", "num_heads", "qk_norm", "use_bias",
                    "context_pre_only", "eps"):
            assert key in cfg

    def test_get_config_round_trip(self):
        layer = MMDiTJointAttention(
            dim=128, num_heads=8, qk_norm=False, use_bias=False,
            context_pre_only=True, eps=1e-5,
        )
        rebuilt = MMDiTJointAttention.from_config(layer.get_config())
        assert rebuilt.dim == 128
        assert rebuilt.num_heads == 8
        assert rebuilt.qk_norm is False
        assert rebuilt.use_bias is False
        assert rebuilt.context_pre_only is True
        assert rebuilt.eps == 1e-5
        assert rebuilt.to_add_out is None

    def _build_model(self, n_img, n_txt, context_pre_only=False):
        img_in = keras.Input(shape=(n_img, self.DIM), name="img")
        txt_in = keras.Input(shape=(n_txt, self.DIM), name="txt")
        out = MMDiTJointAttention(
            dim=self.DIM, num_heads=self.NUM_HEADS,
            context_pre_only=context_pre_only,
        )([img_in, txt_in])
        return keras.Model([img_in, txt_in], out)

    def test_keras_serialization_round_trip(self, sample):
        img, txt = sample
        model = self._build_model(self.N_IMG, self.N_TXT)
        inputs = {
            "img": keras.ops.convert_to_numpy(img),
            "txt": keras.ops.convert_to_numpy(txt),
        }
        out_before = model.predict(inputs, verbose=0)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "mmdit_attn.keras")
            model.save(path)
            reloaded = keras.models.load_model(path)
            out_after = reloaded.predict(inputs, verbose=0)

        # out_before/after are lists (two output streams).
        for a, b in zip(out_before, out_after):
            np.testing.assert_allclose(a, b, atol=1e-6)

    def test_keras_serialization_round_trip_context_pre_only(self, sample):
        img, txt = sample
        model = self._build_model(self.N_IMG, self.N_TXT, context_pre_only=True)
        inputs = {
            "img": keras.ops.convert_to_numpy(img),
            "txt": keras.ops.convert_to_numpy(txt),
        }
        out_before = model.predict(inputs, verbose=0)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "mmdit_attn_cpo.keras")
            model.save(path)
            reloaded = keras.models.load_model(path)
            out_after = reloaded.predict(inputs, verbose=0)
        np.testing.assert_allclose(out_before, out_after, atol=1e-6)

    # ------------------------------------------------------------------
    # Variable batch / variable N_txt
    # ------------------------------------------------------------------

    def test_variable_batch_size(self, layer):
        keras.utils.set_random_seed(3)
        for b in (1, 3, 5):
            img = keras.random.normal((b, self.N_IMG, self.DIM))
            txt = keras.random.normal((b, self.N_TXT, self.DIM))
            img_out, txt_out = layer([img, txt])
            assert tuple(img_out.shape) == (b, self.N_IMG, self.DIM)
            assert tuple(txt_out.shape) == (b, self.N_TXT, self.DIM)

    def test_variable_n_txt(self, layer):
        keras.utils.set_random_seed(5)
        for n_txt in (1, 11, 23):
            img = keras.random.normal((self.BATCH, self.N_IMG, self.DIM))
            txt = keras.random.normal((self.BATCH, n_txt, self.DIM))
            img_out, txt_out = layer([img, txt])
            assert tuple(img_out.shape) == (self.BATCH, self.N_IMG, self.DIM)
            assert tuple(txt_out.shape) == (self.BATCH, n_txt, self.DIM)
