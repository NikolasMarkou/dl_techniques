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

from dl_techniques.models.latent_gmm_registration import (
    LatentGMMRegistration,
    create_latent_gmm_registration,
)

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

    def test_factory_round_trip(self, tmp_path):
        model = create_latent_gmm_registration(num_gaussians=4, k_neighbors=8)
        assert isinstance(model, LatentGMMRegistration)
        clouds = _clouds()
        before = keras.ops.convert_to_numpy(model(clouds, training=False)["estimated_r"])

        path = os.path.join(str(tmp_path), "latent_gmm_factory.keras")
        model.save(path)
        loaded = keras.models.load_model(path)
        after = keras.ops.convert_to_numpy(loaded(clouds, training=False)["estimated_r"])
        np.testing.assert_allclose(before, after, atol=1e-4)


class TestTrainStep:
    """`train_step` was unreachable-by-test and used `keras.backend.GradientTape`,
    which does not exist in Keras 3 -- it raised AttributeError on the first
    optimizer step. These tests walk the custom train/test steps end to end so a
    regression cannot hide behind a green forward-pass-only suite.
    """

    @staticmethod
    def _fit_data(batch=2):
        source, target = _clouds(batch)
        rng = np.random.default_rng(1)
        r_gt = np.tile(np.eye(3, dtype="float32"), (batch, 1, 1))
        t_gt = rng.random((batch, 3)).astype("float32")
        return (source, target), (r_gt, t_gt)

    def test_train_step_runs_and_updates_weights(self):
        model = create_latent_gmm_registration(num_gaussians=4, k_neighbors=8)
        model.compile(optimizer=keras.optimizers.SGD(learning_rate=1e-2))

        (source, target), (r_gt, t_gt) = self._fit_data()
        model((source, target), training=False)
        before = [keras.ops.convert_to_numpy(v) for v in model.trainable_variables]

        logs = model.train_step(((source, target), (r_gt, t_gt)))

        assert "loss" in logs and "chamfer_loss" in logs and "transform_loss" in logs
        assert np.isfinite(float(keras.ops.convert_to_numpy(logs["loss"])))

        after = [keras.ops.convert_to_numpy(v) for v in model.trainable_variables]
        assert any(not np.array_equal(a, b) for a, b in zip(before, after)), \
            "train_step ran but no trainable variable moved"

    def test_train_step_with_compiled_metric(self):
        model = create_latent_gmm_registration(num_gaussians=4, k_neighbors=8)
        model.compile(
            optimizer=keras.optimizers.SGD(learning_rate=1e-2),
            metrics={"estimated_t": keras.metrics.MeanSquaredError(name="t_mse")},
        )
        (source, target), (r_gt, t_gt) = self._fit_data()

        logs = model.train_step(((source, target), (r_gt, t_gt)))
        # Keras prefixes a per-output metric with the OUTPUT name, so a metric
        # named "t_mse" attached to the "estimated_t" output is logged as
        # "estimated_t_t_mse". The model is right; the old assertion named a key
        # Keras never emits and had been red since 3833e8310.
        key = "estimated_t_t_mse"
        assert key in logs, f"expected {key!r} in {sorted(logs)}"
        assert np.isfinite(float(keras.ops.convert_to_numpy(logs[key])))

    def test_test_step_runs(self):
        model = create_latent_gmm_registration(num_gaussians=4, k_neighbors=8)
        model.compile(optimizer=keras.optimizers.SGD(learning_rate=1e-2))
        (source, target), (r_gt, t_gt) = self._fit_data()

        logs = model.test_step(((source, target), (r_gt, t_gt)))
        assert np.isfinite(float(keras.ops.convert_to_numpy(logs["loss"])))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
