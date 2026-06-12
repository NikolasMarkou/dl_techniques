"""Tests for ScalarSinusoidalEmbedding (Ideogram4 scalar/time embedding)."""

import os
import keras
import numpy as np
import pytest

from dl_techniques.layers.embedding.scalar_sinusoidal_embedding import (
    ScalarSinusoidalEmbedding,
    _SINUSOID_SCALE,
)


# ---------------------------------------------------------------------
# numpy reference of the full PyTorch forward (sinusoidal part only)
# ---------------------------------------------------------------------

def _np_sinusoidal(x, dim, range_min, range_max):
    """Numpy reproduction of PyTorch ``_sinusoidal_embedding(scaled, dim)``
    where ``scaled = 1e4 * (x - min) / (max - min)`` (both 1e4 factors).

    Computed in float32 to match the layer's actual dtype: the rescaled
    argument reaches ~1e4, where float64 sin/cos diverges from float32 by
    >1e-6 purely from rounding; the faithful comparison is float32-vs-float32.
    """
    x = np.asarray(x, dtype=np.float32)
    scaled = np.float32(_SINUSOID_SCALE) * (x - np.float32(range_min)) / (
        np.float32(range_max) - np.float32(range_min))
    half = dim // 2
    freq = np.exp(np.arange(half, dtype=np.float32)
                  * -(np.log(_SINUSOID_SCALE) / (half - 1))).astype(np.float32)
    e = (scaled[..., None] * freq).astype(np.float32)
    emb = np.concatenate([np.sin(e), np.cos(e)], axis=-1)
    if dim % 2 == 1:
        pad = np.zeros(emb.shape[:-1] + (1,), dtype=emb.dtype)
        emb = np.concatenate([emb, pad], axis=-1)
    return emb


class TestScalarSinusoidalEmbedding:

    # ---- constructor validation -------------------------------------

    def test_ctor_rejects_bad_range(self):
        with pytest.raises(ValueError):
            ScalarSinusoidalEmbedding(dim=64, input_range=(1.0, 1.0))
        with pytest.raises(ValueError):
            ScalarSinusoidalEmbedding(dim=64, input_range=(1.0, 0.0))

    def test_ctor_rejects_small_dim(self):
        with pytest.raises(ValueError):
            ScalarSinusoidalEmbedding(dim=1, input_range=(0.0, 1.0))

    # ---- sinusoidal numpy reference ---------------------------------

    def test_sinusoidal_matches_numpy_reference(self):
        dim, rmin, rmax = 64, 0.0, 1.0
        layer = ScalarSinusoidalEmbedding(dim=dim, input_range=(rmin, rmax))
        layer.build((None, 1))

        rng = np.random.default_rng(0)
        t = rng.uniform(0.0, 1.0, size=(17,)).astype("float32")

        scaled = _SINUSOID_SCALE * (t - rmin) / (rmax - rmin)
        got = keras.ops.convert_to_numpy(
            layer._sinusoidal(keras.ops.convert_to_tensor(scaled))
        )
        ref = _np_sinusoidal(t, dim, rmin, rmax)
        np.testing.assert_allclose(got, ref, atol=1e-6)

    # ---- output shapes ----------------------------------------------

    @pytest.mark.parametrize("shape", [(8,), (8, 1), (4, 5)])
    def test_output_shape(self, shape):
        dim = 32
        layer = ScalarSinusoidalEmbedding(dim=dim, input_range=(0.0, 1.0))
        x = keras.ops.convert_to_tensor(
            np.random.default_rng(1).uniform(0, 1, size=shape).astype("float32")
        )
        out = layer(x)
        # trailing singleton is squeezed; leading dims preserved
        expected = shape[:-1] + (dim,) if shape[-1] == 1 else shape + (dim,)
        assert tuple(out.shape) == expected

    def test_compute_output_shape(self):
        layer = ScalarSinusoidalEmbedding(dim=48, input_range=(0.0, 1.0))
        assert layer.compute_output_shape((8,)) == (8, 48)
        assert layer.compute_output_shape((8, 1)) == (8, 48)
        assert layer.compute_output_shape((4, 5)) == (4, 5, 48)

    # ---- odd dim padding --------------------------------------------

    def test_odd_dim_padding(self):
        dim = 7
        layer = ScalarSinusoidalEmbedding(dim=dim, input_range=(0.0, 1.0))
        layer.build((None,))
        t = np.array([0.0, 0.3, 0.9], dtype="float32")
        sin = keras.ops.convert_to_numpy(
            layer._sinusoidal(keras.ops.convert_to_tensor(t))
        )
        assert sin.shape == (3, dim)
        # last channel is the pad -> always exactly zero
        np.testing.assert_array_equal(sin[:, -1], np.zeros(3, dtype="float32"))
        # full output also has dim 7
        out = layer(keras.ops.convert_to_tensor(t))
        assert tuple(out.shape) == (3, dim)

    # ---- gradients ---------------------------------------------------

    def test_gradients_finite(self):
        import tensorflow as tf
        layer = ScalarSinusoidalEmbedding(dim=32, input_range=(0.0, 1.0))
        t = tf.convert_to_tensor(
            np.random.default_rng(2).uniform(0, 1, size=(6, 1)).astype("float32")
        )
        with tf.GradientTape() as tape:
            out = layer(t)
            loss = tf.reduce_mean(out ** 2)
        grads = tape.gradient(loss, layer.trainable_variables)
        assert len(grads) == 4  # mlp_in (w,b) + mlp_out (w,b)
        for g in grads:
            assert g is not None
            assert np.all(np.isfinite(keras.ops.convert_to_numpy(g)))

    # ---- .keras round-trip (freq weight must restore) ----------------

    def test_keras_round_trip_restores_freqs(self, tmp_path):
        dim = 40
        inp = keras.Input(shape=(1,), dtype="float32")
        out = ScalarSinusoidalEmbedding(dim=dim, input_range=(0.0, 1.0))(inp)
        model = keras.Model(inp, out)

        t = np.array([[0.0], [0.25], [0.5], [0.75], [1.0]], dtype="float32")
        before = keras.ops.convert_to_numpy(model(t))

        path = os.path.join(tmp_path, "scalar_embed.keras")
        model.save(path)
        reloaded = keras.models.load_model(path)
        after = keras.ops.convert_to_numpy(reloaded(t))

        np.testing.assert_allclose(before, after, atol=1e-6)

        # explicit guard: the freq non-trainable weight restored identically
        orig_layer = next(l for l in model.layers
                          if isinstance(l, ScalarSinusoidalEmbedding))
        new_layer = next(l for l in reloaded.layers
                         if isinstance(l, ScalarSinusoidalEmbedding))
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(orig_layer.freq),
            keras.ops.convert_to_numpy(new_layer.freq),
            atol=1e-6,
        )

        # from_config must restore input_range as a tuple even though the
        # serialized .keras JSON round-trips it to a list.
        rebuilt = ScalarSinusoidalEmbedding.from_config(new_layer.get_config())
        assert isinstance(rebuilt.get_config()["input_range"], tuple)

    def test_get_config_round_trip(self):
        layer = ScalarSinusoidalEmbedding(dim=32, input_range=(-1.0, 2.0))
        cfg = layer.get_config()
        rebuilt = ScalarSinusoidalEmbedding.from_config(cfg)
        assert rebuilt.dim == 32
        assert rebuilt.range_min == -1.0
        assert rebuilt.range_max == 2.0

    def test_from_config_coerces_input_range_to_tuple(self):
        # Simulate JSON having turned the tuple into a list (as .keras does).
        layer = ScalarSinusoidalEmbedding(dim=32, input_range=(-1.0, 2.0))
        cfg = layer.get_config()
        cfg["input_range"] = list(cfg["input_range"])  # tuple -> list
        rebuilt = ScalarSinusoidalEmbedding.from_config(cfg)
        assert isinstance(rebuilt.get_config()["input_range"], tuple)
        assert rebuilt.get_config()["input_range"] == (-1.0, 2.0)
