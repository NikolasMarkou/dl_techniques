"""M2 .keras round-trip + validation tests for the sHGCN model family.

Covers SHGCNModel, SHGCNNodeClassifier, SHGCNLinkPredictor: construction,
ValueError input-validation paths (H4), forward pass, and a full save -> load ->
identical-output round-trip (atol 1e-5).

call() takes an UNBATCHED list [features (N, F), adjacency (N, N)].
"""

import os
import tempfile

import keras
import numpy as np
import pytest
from keras import ops

from dl_techniques.models.shgcn.model import (
    SHGCNModel,
    SHGCNNodeClassifier,
    SHGCNLinkPredictor,
)


def _graph(n=16, f=8):
    features = np.random.rand(n, f).astype("float32")
    adjacency = (np.random.rand(n, n) > 0.5).astype("float32")
    return [features, adjacency]


class TestSHGCNValidation:

    def test_empty_hidden_dims(self):
        with pytest.raises(ValueError, match="at least one dimension"):
            SHGCNModel(hidden_dims=[], output_dim=8)

    def test_nonpositive_hidden_dim(self):
        with pytest.raises(ValueError, match="must be positive"):
            SHGCNModel(hidden_dims=[16, -4], output_dim=8)

    def test_bad_dropout(self):
        with pytest.raises(ValueError, match="dropout_rate"):
            SHGCNModel(hidden_dims=[16], output_dim=8, dropout_rate=1.0)

    def test_classifier_num_classes(self):
        with pytest.raises(ValueError, match="num_classes must be >= 2"):
            SHGCNNodeClassifier(num_classes=1, hidden_dims=[16])

    def test_classifier_delegates_validation(self):
        # hidden_dims validation propagates through the SHGCNModel backbone.
        with pytest.raises(ValueError):
            SHGCNNodeClassifier(num_classes=3, hidden_dims=[])


class TestSHGCNRoundTrip:

    def _run(self, model, atol=1e-5):
        x = _graph()
        y0 = model(x, training=False)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "shgcn.keras")
            model.save(path)
            reloaded = keras.models.load_model(path)
            y1 = reloaded(x, training=False)
        np.testing.assert_allclose(
            ops.convert_to_numpy(y0), ops.convert_to_numpy(y1),
            rtol=1e-5, atol=atol,
        )

    def test_model_round_trip(self):
        self._run(SHGCNModel(hidden_dims=[16, 16], output_dim=8))

    def test_classifier_round_trip(self):
        model = SHGCNNodeClassifier(num_classes=3, hidden_dims=[16, 16])
        out = model(_graph(), training=False)
        assert out.shape == (16, 3)
        self._run(model)

    def test_link_predictor_round_trip(self):
        model = SHGCNLinkPredictor(hidden_dims=[16, 16])
        # link predictor consumes [features, adjacency, edge_pairs]
        features = np.random.rand(16, 8).astype("float32")
        adjacency = (np.random.rand(16, 16) > 0.5).astype("float32")
        edges = np.array([[0, 1], [2, 3], [4, 5]], dtype="int32")
        inp = [features, adjacency, edges]
        y0 = model(inp, training=False)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "shgcn_lp.keras")
            model.save(path)
            reloaded = keras.models.load_model(path)
            y1 = reloaded(inp, training=False)
        np.testing.assert_allclose(
            ops.convert_to_numpy(y0), ops.convert_to_numpy(y1),
            rtol=1e-5, atol=1e-5,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
