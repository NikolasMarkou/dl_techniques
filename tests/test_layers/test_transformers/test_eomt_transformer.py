"""Test suite for EomtTransformer (masked self-attention with object queries)."""
import pytest
import numpy as np
import tensorflow as tf
import keras
import os
import tempfile

from dl_techniques.layers.transformers.eomt_transformer import EomtTransformer


class TestEomtTransformer:
    """Tests for EomtTransformer, focusing on the masked-attention redesign."""

    NUM_PATCHES = 16
    NUM_QUERIES = 4
    H = W = 4  # H * W == NUM_PATCHES
    DIM = 32

    @pytest.fixture
    def x(self):
        return keras.random.normal([2, self.NUM_PATCHES + self.NUM_QUERIES, self.DIM])

    @pytest.fixture
    def seg_mask(self):
        # Segmentation mask in [0, 1], shape (B, num_queries, H, W)
        return keras.ops.cast(
            keras.random.uniform([2, self.NUM_QUERIES, self.H, self.W]) > 0.5, "float32"
        )

    def _layer(self, **kw):
        params = dict(hidden_size=self.DIM, num_heads=4, use_masked_attention=True,
                      mask_probability=1.0, mask_annealing_steps=0)
        params.update(kw)
        return EomtTransformer(**params)

    def test_forward_shape(self, x, seg_mask):
        out = self._layer()(x, mask=seg_mask, training=True)
        assert tuple(out.shape) == (2, self.NUM_PATCHES + self.NUM_QUERIES, self.DIM)

    def test_inference_no_mask(self, x):
        out = self._layer()(x, mask=None, training=False)
        assert out.shape == x.shape
        assert np.all(np.isfinite(np.array(out)))

    def test_masked_attention_changes_output(self, x, seg_mask):
        """The keep-mask must actually reach attention: masked output must differ
        from the unmasked output for the same input."""
        keras.utils.set_random_seed(42)
        layer = self._layer()
        out_masked = np.array(layer(x, mask=seg_mask, training=True))
        out_unmasked = np.array(layer(x, mask=None, training=True))
        assert not np.allclose(out_masked, out_unmasked, atol=1e-5)
        assert np.all(np.isfinite(out_masked))

    def test_empty_mask_no_nan(self, x):
        """An all-zero segmentation mask must not produce NaN (query->query block
        keeps every attention row non-empty)."""
        m0 = keras.ops.zeros([2, self.NUM_QUERIES, self.H, self.W])
        out = self._layer()(x, mask=m0, training=True)
        assert np.all(np.isfinite(np.array(out)))

    def test_graph_trace_training(self, x, seg_mask):
        layer = self._layer()

        @tf.function
        def traced(inp, m):
            return layer(inp, mask=m, training=True)

        out = traced(tf.constant(np.array(x)), tf.constant(np.array(seg_mask)))
        assert tuple(out.shape) == (2, self.NUM_PATCHES + self.NUM_QUERIES, self.DIM)

    def test_get_config_round_trip(self):
        layer = self._layer(mask_annealing_steps=10)
        cfg = layer.get_config()
        rebuilt = EomtTransformer.from_config(cfg)
        assert rebuilt.use_masked_attention is True
        assert rebuilt.mask_probability == 1.0
        assert rebuilt.mask_annealing_steps == 10

    def test_model_save_load_round_trip(self, x, seg_mask):
        inp = keras.Input(shape=(self.NUM_PATCHES + self.NUM_QUERIES, self.DIM))
        out = EomtTransformer(hidden_size=self.DIM, num_heads=4,
                              use_masked_attention=True)(inp)
        model = keras.Model(inp, out)
        ref = model(x, training=False)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "eomt.keras")
            model.save(path)
            loaded = keras.models.load_model(path)
        out2 = loaded(x, training=False)
        np.testing.assert_allclose(np.array(ref), np.array(out2), atol=1e-5)
