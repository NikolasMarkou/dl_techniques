"""
Construction + M2 round-trip test for RelGT (relational graph transformer).

create_relgt_model(output_dim, model_size) takes a dict of graph tensors and
returns (B, output_dim). NOTE: RelGT performs stochastic local-neighborhood
sampling, so its forward is NOT deterministic even at inference — output-identity
across a save/load cannot be asserted. M2 is therefore verified by confirming
ALL weights are preserved (by order) across the .keras round-trip.
"""

import os
import keras
import pytest
import numpy as np

from dl_techniques.models.relgt.model import create_relgt_model

B, N, F = 2, 8, 16


def _inputs():
    rng = np.random.default_rng(0)
    return {
        "node_features": rng.random((B, N, F)).astype("float32"),
        "node_types": rng.integers(0, 10, (B, N)).astype("int32"),
        "hop_distances": rng.integers(0, 3, (B, N)).astype("int32"),
        "relative_times": rng.random((B, N, 1)).astype("float32"),
        "subgraph_adjacency": rng.random((B, N, N)).astype("float32"),
    }


def _model():
    return create_relgt_model(output_dim=2, model_size="small")


class TestRelGT:

    def test_forward_shape(self):
        out = _model()(_inputs(), training=False)
        assert tuple(out.shape) == (B, 2)
        assert not np.any(np.isnan(keras.ops.convert_to_numpy(out)))

    def test_keras_round_trip_weights_preserved(self, tmp_path):
        model = _model()
        x = _inputs()
        _ = model(x, training=False)  # build

        path = os.path.join(str(tmp_path), "relgt.keras")
        model.save(path)
        loaded = keras.models.load_model(path)
        _ = loaded(x, training=False)

        assert len(model.weights) == len(loaded.weights)
        # Forward is stochastic -> compare weights (by order) instead of outputs.
        for w_orig, w_load in zip(model.weights, loaded.weights):
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(w_orig),
                keras.ops.convert_to_numpy(w_load),
                atol=1e-6,
                err_msg=f"weight {w_orig.path} not preserved after round-trip",
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
