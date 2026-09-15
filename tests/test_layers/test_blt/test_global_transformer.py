"""Tests for ``GlobalTransformer`` (``dl_techniques.layers.blt.global_transformer``)."""

import os

import keras
import numpy as np

from dl_techniques.layers.blt.global_transformer import GlobalTransformer

NP_, GDIM = 8, 16
B = 2


# ---------------------------------------------------------------------
# GlobalTransformer (single-tensor call -> functional .keras round-trip)
# ---------------------------------------------------------------------

class TestGlobalTransformer:

    def _make(self):
        return GlobalTransformer(global_dim=GDIM, num_global_layers=1,
                                num_heads_global=2, max_patches=NP_)

    def test_forward_pass(self):
        x = keras.ops.convert_to_tensor(
            np.random.default_rng(0).standard_normal((B, NP_, GDIM)).astype("float32")
        )
        assert tuple(self._make()(x).shape) == (B, NP_, GDIM)

    def test_compute_output_shape(self):
        assert self._make().compute_output_shape((B, NP_, GDIM)) == (B, NP_, GDIM)

    def test_serialization_round_trip(self, tmp_path):
        inp = keras.Input(shape=(NP_, GDIM))
        out = self._make()(inp)
        model = keras.Model(inp, out)
        x = np.random.default_rng(0).standard_normal((B, NP_, GDIM)).astype("float32")
        y0 = model(x)
        path = os.path.join(tmp_path, "global.keras")
        model.save(path)
        loaded = keras.models.load_model(
            path, custom_objects={"GlobalTransformer": GlobalTransformer}
        )
        y1 = loaded(x)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(y0), keras.ops.convert_to_numpy(y1),
            rtol=1e-5, atol=1e-5,
        )
