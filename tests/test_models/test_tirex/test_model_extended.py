"""M2 .keras round-trip test for TiRexExtended (query-token TiRex variant).

The audit scanner mislabeled tirex/model_extended.py as N/A, but TiRexExtended is
a concrete keras.Model subclass (of TiRexCore) with a build() override and its own
query-token / token-wise-head topology. This test gives it real coverage:
construction, forward shape, and a full save -> load -> identical-output round-trip.
"""

import os
import tempfile

import keras
import numpy as np
import pytest
from keras import ops

from dl_techniques.models.time_series.tirex.model_extended import (
    TiRexExtended,
    create_tirex_extended,
)


def _config():
    return dict(
        patch_size=8,
        embed_dim=32,
        num_blocks=2,
        num_heads=4,
        lstm_units=32,
        ff_dim=64,
        block_types=['mixed', 'lstm'],
        quantile_levels=[0.1, 0.5, 0.9],
        prediction_length=12,
        dropout_rate=0.1,
    )


def _series(b=4, seq=64):
    return keras.random.normal(shape=(b, seq, 1))


class TestTiRexExtended:

    def test_forward_shape(self):
        model = TiRexExtended(**_config())
        out = model(_series(), training=False)
        # (B, prediction_length, num_quantiles)
        assert out.shape == (4, 12, 3)

    def test_factory(self):
        model = create_tirex_extended(
            "small", input_length=64, prediction_length=12
        )
        assert isinstance(model, TiRexExtended)

    def test_keras_round_trip(self):
        model = TiRexExtended(**_config())
        x = _series()
        y0 = model(x, training=False)

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "tirex_ext.keras")
            model.save(path)
            reloaded = keras.models.load_model(path)
            y1 = reloaded(x, training=False)

        assert reloaded.prediction_length == model.prediction_length
        np.testing.assert_allclose(
            ops.convert_to_numpy(y0), ops.convert_to_numpy(y1),
            rtol=1e-5, atol=1e-5,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
