"""Tests for the DeepAR likelihood heads."""

import os
import keras
import numpy as np

from dl_techniques.layers.heads.time_series.likelihood_heads import (
    GaussianLikelihoodHead,
    NegativeBinomialLikelihoodHead,
)

B, T, D = 4, 6, 3


def _f32(*shape):
    return keras.ops.convert_to_tensor(
        np.random.default_rng(0).standard_normal(shape).astype("float32")
    )


class TestGaussianLikelihoodHead:
    def test_forward_and_shape(self):
        head = GaussianLikelihoodHead(units=2)
        mu, sigma = head(_f32(B, T, D))
        assert tuple(mu.shape) == (B, T, 2) and tuple(sigma.shape) == (B, T, 2)
        m_shape, s_shape = head.compute_output_shape((B, T, D))
        assert m_shape == (B, T, 2) and s_shape == (B, T, 2)
        assert np.all(keras.ops.convert_to_numpy(sigma) > 0)  # sigma is positive

    def test_serialization(self, tmp_path):
        inp = keras.Input(shape=(T, D))
        mu, sigma = GaussianLikelihoodHead(units=2, name="gh")(inp)
        model = keras.Model(inp, [mu, sigma])
        x = np.random.default_rng(0).standard_normal((B, T, D)).astype("float32")
        m0, s0 = model(x)
        path = os.path.join(tmp_path, "gh.keras")
        model.save(path)
        loaded = keras.models.load_model(
            path, custom_objects={"GaussianLikelihoodHead": GaussianLikelihoodHead}
        )
        m1, s1 = loaded(x)
        np.testing.assert_allclose(keras.ops.convert_to_numpy(m0), keras.ops.convert_to_numpy(m1), rtol=1e-5, atol=1e-5)


class TestNegativeBinomialLikelihoodHead:
    def test_forward(self):
        head = NegativeBinomialLikelihoodHead(units=2)
        out = head(_f32(B, T, D))
        assert isinstance(out, tuple) and len(out) == 2
        for t in out:
            assert tuple(t.shape) == (B, T, 2)
