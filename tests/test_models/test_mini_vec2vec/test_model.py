"""
Test suite for MiniVec2VecAligner.

The aligner applies a single learned linear transform W to one embedding tensor.
call() takes a (B, embedding_dim) float32 input and returns ops.matmul(inputs, W)
of the same shape (W is identity-initialized in build()). Covers construction
(incl. ValueError), a forward pass, and the M2 full .keras round-trip.
"""

import os
import keras
import pytest
import numpy as np

from dl_techniques.models.mini_vec2vec.model import (
    MiniVec2VecAligner, create_mini_vec2vec_aligner,
)

DIM = 128


def _emb(batch=2):
    return np.random.default_rng(0).random((batch, DIM)).astype("float32")


class TestMiniVec2VecAligner:

    def test_factory_construction(self):
        model = create_mini_vec2vec_aligner(embedding_dim=DIM)
        assert isinstance(model, MiniVec2VecAligner)

    def test_invalid_embedding_dim_raises(self):
        with pytest.raises(ValueError, match="embedding_dim"):
            MiniVec2VecAligner(embedding_dim=0)

    def test_forward_shape(self):
        model = create_mini_vec2vec_aligner(embedding_dim=DIM)
        x = _emb()
        y = keras.ops.convert_to_numpy(model(x, training=False))
        assert y.shape == (2, DIM)
        assert not np.any(np.isnan(y))

    def test_keras_round_trip(self, tmp_path):
        model = create_mini_vec2vec_aligner(embedding_dim=DIM)
        x = _emb()
        before = keras.ops.convert_to_numpy(model(x, training=False))

        path = os.path.join(str(tmp_path), "mini_vec2vec.keras")
        model.save(path)
        loaded = keras.models.load_model(path)
        after = keras.ops.convert_to_numpy(loaded(x, training=False))

        # GPU fp32 reduction noise -> atol 1e-4 (SYSTEM invariant)
        np.testing.assert_allclose(before, after, atol=1e-4,
                                   err_msg="Aligner differs after .keras round-trip")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
