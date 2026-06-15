"""Tests for PositionEmbeddingSine2D (fixed 2D sinusoidal positional encoding)."""

import os
import numpy as np
import pytest
import keras
import tensorflow as tf

from dl_techniques.layers.embedding.positional_embedding_sine_2d import PositionEmbeddingSine2D


class TestPositionEmbeddingSine2D:

    # ---- constructor validation -------------------------------------

    def test_ctor_rejects_bad_args(self):
        with pytest.raises(ValueError):
            PositionEmbeddingSine2D(num_pos_feats=0)
        with pytest.raises(ValueError):
            PositionEmbeddingSine2D(temperature=0.0)

    # ---- forward / shape (channels-first NCHW) ----------------------

    def test_forward_shape_channels_first(self):
        layer = PositionEmbeddingSine2D(num_pos_feats=8)
        x = keras.ops.convert_to_tensor(np.random.rand(2, 6, 5, 3).astype("float32"))
        out = layer(x)
        # (B, 2*num_pos_feats, H, W)
        assert tuple(out.shape) == (2, 16, 6, 5)

    def test_compute_output_shape(self):
        layer = PositionEmbeddingSine2D(num_pos_feats=8)
        assert layer.compute_output_shape((2, 6, 5, 3)) == (2, 16, 6, 5)

    def test_normalize_false(self):
        layer = PositionEmbeddingSine2D(num_pos_feats=8, normalize=False)
        x = keras.ops.convert_to_tensor(np.random.rand(2, 6, 5, 3).astype("float32"))
        out = keras.ops.convert_to_numpy(layer(x))
        assert out.shape == (2, 16, 6, 5)
        assert np.all(np.isfinite(out))

    # ---- graph safety -----------------------------------------------

    def test_graph_trace(self):
        layer = PositionEmbeddingSine2D(num_pos_feats=8)
        x = tf.constant(np.random.rand(2, 6, 5, 3).astype("float32"))
        eager = keras.ops.convert_to_numpy(layer(x))
        f = tf.function(lambda t: layer(t),
                        input_signature=[tf.TensorSpec([None, None, None, 3], tf.float32)])
        np.testing.assert_allclose(eager, f(x).numpy(), atol=1e-6)

    # ---- serialization ----------------------------------------------

    def test_get_config_round_trip(self):
        layer = PositionEmbeddingSine2D(num_pos_feats=16, temperature=5000.0, normalize=False)
        rebuilt = PositionEmbeddingSine2D.from_config(layer.get_config())
        assert rebuilt.num_pos_feats == 16
        assert rebuilt.temperature == 5000.0
        assert rebuilt.normalize is False

    def test_keras_round_trip(self, tmp_path):
        inp = keras.Input(shape=(6, 5, 3), dtype="float32")
        out = PositionEmbeddingSine2D(num_pos_feats=8)(inp)
        model = keras.Model(inp, out)
        x = np.random.rand(2, 6, 5, 3).astype("float32")
        before = keras.ops.convert_to_numpy(model(x))
        path = os.path.join(tmp_path, "pe2d.keras")
        model.save(path)
        after = keras.ops.convert_to_numpy(keras.models.load_model(path)(x))
        np.testing.assert_allclose(before, after, atol=1e-6)
