"""M2 .keras round-trip + validation tests for the TabM model.

Covers: construction, ValueError input-validation paths (H4 — now raised as
ValueError, not assert), forward pass, and a full save -> load -> identical-output
round-trip (atol 1e-6).
"""

import os
import tempfile

import keras
import numpy as np
import pytest
from keras import ops

from dl_techniques.models.tabm.model import TabMModel, create_tabm_mini


def _features(b=4, n=8):
    return np.random.rand(b, n).astype("float32")


class TestTabMValidation:
    """H4: invalid __init__ args raise ValueError (not AssertionError)."""

    def test_empty_features(self):
        with pytest.raises(ValueError, match="either numerical or categorical"):
            TabMModel(n_num_features=0, cat_cardinalities=[], n_classes=3,
                      hidden_dims=[16])

    def test_negative_num_features(self):
        with pytest.raises(ValueError, match="n_num_features must be non-negative"):
            TabMModel(n_num_features=-1, cat_cardinalities=[], n_classes=3,
                      hidden_dims=[16])

    def test_empty_hidden_dims(self):
        with pytest.raises(ValueError, match="hidden_dims cannot be empty"):
            TabMModel(n_num_features=8, cat_cardinalities=[], n_classes=3,
                      hidden_dims=[])

    def test_bad_dropout(self):
        with pytest.raises(ValueError, match="dropout_rate must be in"):
            TabMModel(n_num_features=8, cat_cardinalities=[], n_classes=3,
                      hidden_dims=[16], dropout_rate=1.5)

    def test_ensemble_requires_k(self):
        with pytest.raises(ValueError, match="requires k to be specified"):
            TabMModel(n_num_features=8, cat_cardinalities=[], n_classes=3,
                      hidden_dims=[16], arch_type='tabm', k=None)

    def test_bad_cardinalities(self):
        with pytest.raises(ValueError, match="cardinalities must be positive"):
            TabMModel(n_num_features=0, cat_cardinalities=[3, 0], n_classes=3,
                      hidden_dims=[16])


class TestTabMRoundTrip:

    def test_forward_shape(self):
        model = create_tabm_mini(n_num_features=8, cat_cardinalities=[],
                                 n_classes=3)
        out = model(_features(), training=False)
        # ensemble output (B, k, n_classes)
        assert out.shape[0] == 4
        assert out.shape[-1] == 3

    def test_keras_round_trip(self):
        model = create_tabm_mini(n_num_features=8, cat_cardinalities=[],
                                 n_classes=3)
        x = _features()
        y0 = model(x, training=False)

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "tabm.keras")
            model.save(path)
            reloaded = keras.models.load_model(path)
            y1 = reloaded(x, training=False)

        np.testing.assert_allclose(
            ops.convert_to_numpy(y0), ops.convert_to_numpy(y1),
            rtol=1e-6, atol=1e-6,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ---------------------------------------------------------------------
# Gradient flow (plan-2026-08-19-a616f581 step 10)
# ---------------------------------------------------------------------

from ..gradient_flow_oracle import assert_gradients_reach_every_trainable_weight


class TestTabMGradientFlow:
    """Every trainable weight must be on the backward graph.

    TabM is an ENSEMBLE model: ``call()`` returns ``(B, k, n_classes)`` and the
    per-member adapters are the whole point of the architecture. A dead adapter
    is invisible to every shape and finiteness test in this suite -- and the
    ``arch_type`` collapse this package already suffered once
    (``test_arch_types.py``, three arch_types building byte-identical models)
    is exactly the family of defect a per-weight gradient claim detects.
    """

    def test_gradients_reach_every_trainable_weight(self):
        model = create_tabm_mini(
            n_num_features=8, cat_cardinalities=[], n_classes=3
        )
        x = _features()
        model(x, training=False)  # a subclassed model is unbuilt until first call

        report = assert_gradients_reach_every_trainable_weight(model, x)

        assert len(report) == len(model.trainable_weights)
        assert len(report) > 0
        assert max(v for v in report.values() if v is not None) > 0.0
