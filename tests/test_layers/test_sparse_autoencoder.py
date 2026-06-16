"""Tests for the SparseAutoencoder layer."""

import os
import keras
import numpy as np
import pytest

from dl_techniques.layers.sparse_autoencoder import SparseAutoencoder

B, DIN, DLAT = 4, 10, 20


@pytest.fixture
def sample():
    return np.random.default_rng(0).standard_normal((B, DIN)).astype("float32")


class TestSparseAutoencoder:

    def test_construction(self):
        layer = SparseAutoencoder(d_input=DIN, d_latent=DLAT, variant="topk", k=4)
        assert layer.d_latent == DLAT

    def test_invalid_variant(self):
        with pytest.raises(ValueError):
            SparseAutoencoder(d_input=DIN, d_latent=DLAT, variant="bogus")

    def test_topk_requires_k(self):
        with pytest.raises(ValueError):
            SparseAutoencoder(d_input=DIN, d_latent=DLAT, variant="topk", k=None)

    @pytest.mark.parametrize("variant,kw", [
        ("relu", {}),
        ("topk", {"k": 4}),
        ("batch_topk", {"k": 4}),
        ("jumprelu", {}),
        ("gated", {}),
    ])
    def test_forward_pass(self, sample, variant, kw):
        layer = SparseAutoencoder(d_input=DIN, d_latent=DLAT, variant=variant, **kw)
        out = layer(sample)
        assert tuple(out.shape) == (B, DIN)
        assert np.all(np.isfinite(keras.ops.convert_to_numpy(out)))

    def test_return_latents(self, sample):
        layer = SparseAutoencoder(d_input=DIN, d_latent=DLAT, variant="topk", k=4)
        recon, latents, loss = layer(sample, return_latents=True)
        assert tuple(recon.shape) == (B, DIN)
        assert tuple(latents.shape) == (B, DLAT)

    def test_compute_output_shape(self):
        layer = SparseAutoencoder(d_input=DIN, d_latent=DLAT, variant="topk", k=4)
        assert layer.compute_output_shape((B, DIN)) == (B, DIN)

    def test_serialization_round_trip(self, sample, tmp_path):
        inp = keras.Input(shape=(DIN,))
        out = SparseAutoencoder(d_input=DIN, d_latent=DLAT, variant="topk", k=4, name="sae")(inp)
        model = keras.Model(inp, out)
        y0 = model(sample, training=False)
        path = os.path.join(tmp_path, "sae.keras")
        model.save(path)
        loaded = keras.models.load_model(
            path, custom_objects={"SparseAutoencoder": SparseAutoencoder}
        )
        y1 = loaded(sample, training=False)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(y0), keras.ops.convert_to_numpy(y1),
            rtol=1e-5, atol=1e-5,
        )

    def test_get_config_round_trip(self):
        layer = SparseAutoencoder(d_input=DIN, d_latent=DLAT, variant="topk", k=4, tied_weights=True)
        rebuilt = SparseAutoencoder.from_config(layer.get_config())
        assert rebuilt.d_latent == DLAT and rebuilt.variant == "topk"
