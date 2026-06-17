"""
Test suite for LatentGMMRegistration (point cloud registration via latent GMM).

Covers construction, a forward pass, and the M2 full .keras
save -> load -> identical-output round-trip. The model's weighted-Procrustes
solver uses a documented raw-tf SVD path (accepted §L2-5 exception); the model
still serializes and round-trips cleanly.

Input is a tuple (source_pc, target_pc), each (B, N, 3). Output is a dict with
reconstruction_x/y + estimated_r (rotation) + estimated_t (translation).
"""

import os
import keras
import pytest
import numpy as np

from dl_techniques.models.latent_gmm_registration.model import LatentGMMRegistration

N_POINTS = 64


def _model():
    return LatentGMMRegistration(num_gaussians=4, k_neighbors=8)


def _clouds(batch=2):
    rng = np.random.default_rng(0)
    return (
        rng.random((batch, N_POINTS, 3)).astype("float32"),
        rng.random((batch, N_POINTS, 3)).astype("float32"),
    )


class TestLatentGMMRegistration:

    def test_forward_dict(self):
        out = _model()(_clouds(), training=False)
        assert {"reconstruction_x", "reconstruction_y",
                "estimated_r", "estimated_t"} <= set(out)
        assert tuple(out["estimated_r"].shape) == (2, 3, 3)
        for v in out.values():
            assert not np.any(np.isnan(keras.ops.convert_to_numpy(v)))

    def test_keras_round_trip(self, tmp_path):
        model = _model()
        clouds = _clouds()
        # estimated_r flows through the raw-tf SVD Procrustes solver
        before = keras.ops.convert_to_numpy(model(clouds, training=False)["estimated_r"])

        path = os.path.join(str(tmp_path), "latent_gmm.keras")
        model.save(path)
        loaded = keras.models.load_model(path)
        after = keras.ops.convert_to_numpy(loaded(clouds, training=False)["estimated_r"])

        # GPU fp32 reduction noise -> atol 1e-4 (SYSTEM invariant)
        np.testing.assert_allclose(before, after, atol=1e-4,
                                   err_msg="LatentGMMRegistration differs after round-trip")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
