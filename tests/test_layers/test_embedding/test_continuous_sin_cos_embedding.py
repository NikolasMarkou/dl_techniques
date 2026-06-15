"""Tests for ContinuousSinCosEmbed (fixed sin/cos embedding of continuous coords)."""

import os
import numpy as np
import pytest
import keras
import tensorflow as tf

from dl_techniques.layers.embedding.continuous_sin_cos_embedding import ContinuousSinCosEmbed


class TestContinuousSinCosEmbed:

    # ---- constructor validation -------------------------------------

    def test_ctor_rejects_bad_args(self):
        with pytest.raises(ValueError):
            ContinuousSinCosEmbed(dim=0, ndim=2)
        with pytest.raises(ValueError):
            ContinuousSinCosEmbed(dim=64, ndim=0)
        with pytest.raises(ValueError):
            ContinuousSinCosEmbed(dim=64, ndim=2, max_wavelength=0.0)
        with pytest.raises(ValueError):
            ContinuousSinCosEmbed(dim=2, ndim=4)

    # ---- forward / shape --------------------------------------------

    @pytest.mark.parametrize("shape,ndim", [((2, 5, 3), 3), ((5, 3), 3), ((2, 7, 2), 2)])
    def test_forward_shape(self, shape, ndim):
        layer = ContinuousSinCosEmbed(dim=64, ndim=ndim)
        x = keras.ops.convert_to_tensor(np.random.rand(*shape).astype("float32"))
        out = layer(x)
        # full embedding width == dim
        assert tuple(out.shape) == tuple(shape[:-1]) + (64,)

    def test_compute_output_shape_matches_actual(self):
        for dim, ndim, in_shape in [(64, 3, (2, 5, 3)), (66, 4, (2, 5, 4)), (60, 3, (2, 5, 3))]:
            layer = ContinuousSinCosEmbed(dim=dim, ndim=ndim)
            x = keras.ops.convert_to_tensor(np.random.rand(*in_shape).astype("float32"))
            actual = int(layer(x).shape[-1])
            declared = layer.compute_output_shape(in_shape)[-1]
            assert actual == declared == dim

    # ---- graph safety (locks the removed eager convert_to_numpy) -----

    def test_graph_trace_no_eager(self):
        layer = ContinuousSinCosEmbed(dim=64, ndim=3)  # assert_positive=True default
        x = tf.constant(np.random.rand(2, 5, 3).astype("float32"))
        eager = keras.ops.convert_to_numpy(layer(x))
        f = tf.function(lambda t: layer(t),
                        input_signature=[tf.TensorSpec([None, None, 3], tf.float32)])
        np.testing.assert_allclose(eager, f(x).numpy(), atol=1e-6)

    def test_graph_trace_with_padding(self):
        layer = ContinuousSinCosEmbed(dim=66, ndim=4)
        x = tf.constant(np.random.rand(2, 5, 4).astype("float32"))
        f = tf.function(lambda t: layer(t),
                        input_signature=[tf.TensorSpec([None, None, 4], tf.float32)])
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(layer(x)), f(x).numpy(), atol=1e-6)

    # ---- build idempotency ------------------------------------------

    def test_double_build_stable(self):
        layer = ContinuousSinCosEmbed(dim=64, ndim=3)
        layer.build((2, 5, 3))
        n1 = len(layer.weights)
        layer.build((2, 5, 3))
        assert len(layer.weights) == n1

    # ---- serialization ----------------------------------------------

    def test_get_config_round_trip(self):
        layer = ContinuousSinCosEmbed(dim=48, ndim=3, max_wavelength=5000.0, assert_positive=False)
        rebuilt = ContinuousSinCosEmbed.from_config(layer.get_config())
        assert rebuilt.dim == 48 and rebuilt.ndim == 3
        assert rebuilt.max_wavelength == 5000.0 and rebuilt.assert_positive is False

    def test_keras_round_trip(self, tmp_path):
        inp = keras.Input(shape=(5, 3), dtype="float32")
        out = ContinuousSinCosEmbed(dim=64, ndim=3)(inp)
        model = keras.Model(inp, out)
        x = np.random.rand(2, 5, 3).astype("float32")
        before = keras.ops.convert_to_numpy(model(x))
        path = os.path.join(tmp_path, "csincos.keras")
        model.save(path)
        after = keras.ops.convert_to_numpy(keras.models.load_model(path)(x))
        np.testing.assert_allclose(before, after, atol=1e-6)
