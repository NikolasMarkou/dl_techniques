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

from dl_techniques.models.graph.shgcn.model import (
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

    def test_keras_roundtrip_bit_for_bit_with_string_activation(self):
        """New .keras save/load round-trip test (plan Item 5b).

        Builds `SHGCNModel(hidden_dims=[16, 16], output_dim=8,
        output_activation='tanh')` -- a non-default `output_activation`
        (the class default is `'linear'`) -- runs a deterministic forward
        pass over `_graph()`'s fixed-shape graph at `training=False`,
        `.save()`s to a `tempfile.TemporaryDirectory()`-backed `.keras`
        path, `keras.models.load_model()`s it back, and compares the
        reloaded forward pass to the original BIT-FOR-BIT
        (`np.testing.assert_array_equal`, not `assert_allclose`). This is a
        stronger claim than `test_model_round_trip` above (which uses
        `rtol=1e-5, atol=1e-5` and the DEFAULT `'linear'` activation) --
        this test additionally proves the file-based save/load mechanism
        survives a non-default, non-linear `output_activation` string with
        ZERO drift, not just drift under a tolerance.

        RED-proof: see
        `test_keras_roundtrip_detects_weight_perturbation_with_string_activation`
        below, a permanent second test that perturbs the reloaded model's
        weights by a known amount and asserts the bit-for-bit comparison
        DOES fail.
        """
        model = SHGCNModel(hidden_dims=[16, 16], output_dim=8, output_activation='tanh')
        x = _graph()
        original_prediction = ops.convert_to_numpy(model(x, training=False))

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "shgcn_tanh.keras")
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_prediction = ops.convert_to_numpy(loaded_model(x, training=False))

        np.testing.assert_array_equal(
            original_prediction,
            loaded_prediction,
            err_msg="Reloaded model's forward pass is not bit-for-bit identical",
        )

    def test_keras_roundtrip_detects_weight_perturbation_with_string_activation(self):
        """RED-proof for `test_keras_roundtrip_bit_for_bit_with_string_activation`.

        Repeats the same save/load round trip with `output_activation='tanh'`,
        then perturbs the reloaded model's output-layer kernel by a known,
        clearly-detectable amount before comparing. Asserts the bit-for-bit
        comparison DOES raise `AssertionError` against the perturbed
        reload, proving the comparison above is not vacuously passing.
        """
        model = SHGCNModel(hidden_dims=[16, 16], output_dim=8, output_activation='tanh')
        x = _graph()
        original_prediction = ops.convert_to_numpy(model(x, training=False))

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "shgcn_tanh_perturbed.keras")
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)

            output_layer = loaded_model.output_layer
            weights = output_layer.get_weights()
            weights[0] = weights[0] + 1.0
            output_layer.set_weights(weights)

            perturbed_prediction = ops.convert_to_numpy(loaded_model(x, training=False))

        with pytest.raises(AssertionError):
            np.testing.assert_array_equal(
                original_prediction,
                perturbed_prediction,
                err_msg="Perturbation should have been detected but was not",
            )

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
