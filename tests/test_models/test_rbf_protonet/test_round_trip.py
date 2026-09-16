"""
``get_config``/``from_config`` and ``.keras`` save/load round-trip tests for
``RBFProtoNet``, following ``tests/test_models/test_resnet/test_round_trip.py``
conventions.

Both round trips must reproduce the original model's output on a fixed input
within a small tolerance -- the `.keras` round trip through weight
serialization is expected to be bit-identical (single-device CPU, no
reduction-order nondeterminism to absorb), so it is pinned tighter than the
GPU fp32 tolerance ResNet's own round-trip test uses.
"""

import os
import keras
import numpy as np
import pytest

from dl_techniques.models.vision.rbf_protonet.model import RBFProtoNet


def _model(**overrides) -> RBFProtoNet:
    cfg = dict(input_shape=(32, 32, 3), num_classes=10)
    cfg.update(overrides)
    return RBFProtoNet(**cfg)


def _images(batch: int = 2) -> np.ndarray:
    return np.random.default_rng(0).random((batch, 32, 32, 3)).astype("float32")


class TestRBFProtoNetConfigRoundTrip:
    def test_get_config_carries_every_constructor_arg(self):
        model = _model(
            stem_filters=16,
            filters_per_stage=[32, 64],
            feature_dim=64,
            num_classes=10,
            repulsion_strength=0.2,
            min_distance=0.5,
        )
        cfg = model.get_config()
        assert cfg["stem_filters"] == 16
        assert cfg["filters_per_stage"] == [32, 64]
        assert cfg["feature_dim"] == 64
        assert cfg["num_classes"] == 10
        assert cfg["repulsion_strength"] == 0.2
        assert cfg["min_distance"] == 0.5
        assert cfg["pretrained"] is False

    def test_from_config_rebuild_matches_original_output(self):
        """A model rebuilt from `get_config()` reproduces the original's output
        within `atol=1e-5`, PROVIDED it is built from the same weights -- a
        fresh `from_config()` call draws fresh random init, so weights are
        copied across explicitly before comparing (the round trip under test
        is the CONFIG plumbing, not re-initialization luck)."""
        model = _model()
        x = _images()
        # Build both models by tracing one forward pass each.
        _ = model(x, training=False)

        rebuilt = RBFProtoNet.from_config(model.get_config())
        _ = rebuilt(x, training=False)
        rebuilt.set_weights(model.get_weights())

        before = keras.ops.convert_to_numpy(model(x, training=False))
        after = keras.ops.convert_to_numpy(rebuilt(x, training=False))
        np.testing.assert_allclose(
            before, after, atol=1e-5, rtol=0,
            err_msg="RBFProtoNet.from_config() rebuild differs from the original after weight copy",
        )


class TestRBFProtoNetKerasRoundTrip:
    def test_forward_shape(self):
        out = _model()(_images(), training=False)
        assert tuple(out.shape) == (2, 10)

    def test_keras_save_load_round_trip(self, tmp_path):
        model = _model()
        x = _images()
        before = keras.ops.convert_to_numpy(model(x, training=False))

        path = os.path.join(str(tmp_path), "rbf_protonet.keras")
        model.save(path)
        loaded = keras.models.load_model(path)
        after = keras.ops.convert_to_numpy(loaded(x, training=False))

        np.testing.assert_allclose(
            before, after, atol=1e-5, rtol=0,
            err_msg="RBFProtoNet differs after .keras round-trip",
        )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-vvv"]))
