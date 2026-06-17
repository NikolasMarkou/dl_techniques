"""
Round-trip + construction tests for DINOv1 and its DINOHead sub-layer.

Complements the forward-only smoke (test_dino_v1.py). The DINOHead and the full
DINOv1 .keras round-trips were previously broken (DINOHead sublayers built
lazily; DINOv1 patch_size/image_size deserialized as lists breaking `//`). Both
are fixed and pinned here with M2 identical-output assertions.
"""

import os
import keras
import pytest
import numpy as np

from dl_techniques.models.dino.dino_v1 import DINOHead, create_dino_v1


class TestDINOHead:

    def test_forward_shape(self):
        head = DINOHead(in_dim=64, out_dim=128, nlayers=3,
                        hidden_dim=256, bottleneck_dim=64)
        x = np.random.default_rng(0).random((4, 64)).astype("float32")
        y = head(x, training=False)
        assert tuple(y.shape) == (4, 128)

    def test_keras_round_trip(self, tmp_path):
        head = DINOHead(in_dim=64, out_dim=128, nlayers=3,
                        hidden_dim=256, bottleneck_dim=64)
        model = keras.Sequential([keras.Input((64,)), head])
        x = np.random.default_rng(1).random((4, 64)).astype("float32")
        before = keras.ops.convert_to_numpy(model(x, training=False))

        path = os.path.join(str(tmp_path), "dino_head.keras")
        model.save(path)
        loaded = keras.models.load_model(path)
        after = keras.ops.convert_to_numpy(loaded(x, training=False))
        np.testing.assert_allclose(before, after, atol=1e-4,
                                   err_msg="DINOHead differs after round-trip")


class TestDINOv1:

    def _model(self):
        return create_dino_v1("small", num_classes=10, patch_size=16,
                              input_shape=(32, 32, 3))

    def test_forward_logits(self):
        model = self._model()
        x = np.random.default_rng(2).random((2, 32, 32, 3)).astype("float32")
        out = model(x, training=False)
        assert tuple(out.shape) == (2, 10)

    def test_keras_round_trip(self, tmp_path):
        model = self._model()
        x = np.random.default_rng(3).random((2, 32, 32, 3)).astype("float32")
        before = keras.ops.convert_to_numpy(model(x, training=False))

        path = os.path.join(str(tmp_path), "dino_v1.keras")
        model.save(path)
        loaded = keras.models.load_model(path)
        after = keras.ops.convert_to_numpy(loaded(x, training=False))
        # GPU fp32 reduction noise -> atol 1e-4 (SYSTEM invariant)
        np.testing.assert_allclose(before, after, atol=1e-4,
                                   err_msg="DINOv1 differs after .keras round-trip")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
