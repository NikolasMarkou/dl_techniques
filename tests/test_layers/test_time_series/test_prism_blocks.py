"""Tests for the PRISM time-series blocks.

The top-level ``PRISMLayer`` composes ``PRISMTimeTree`` / ``PRISMNode`` /
``FrequencyBandRouter`` / ``FrequencyBandStatistics``, so its forward pass and
``.keras`` round-trip exercise those sub-layers. ``FrequencyBandStatistics`` is
also tested directly.
"""

import os
import keras
import numpy as np
import pytest

from dl_techniques.layers.time_series.prism_blocks import (
    PRISMLayer,
    FrequencyBandStatistics,
)

# seq_len must be long enough for the multi-level wavelet decomposition not to
# degenerate (very short bands collapse the band-length normalization).
B, SEQ, CH = 4, 32, 4


@pytest.fixture
def sample():
    return np.random.default_rng(0).standard_normal((B, SEQ, CH)).astype("float32")


class TestFrequencyBandStatistics:
    def test_forward_and_shape(self, sample):
        layer = FrequencyBandStatistics()
        out = layer(sample)
        n_stats = layer._num_stats
        assert tuple(out.shape) == (B, CH, n_stats)
        assert layer.compute_output_shape((B, SEQ, CH)) == (B, CH, n_stats)

    def test_serialization(self, sample, tmp_path):
        inp = keras.Input(shape=(SEQ, CH))
        out = FrequencyBandStatistics(name="fbs")(inp)
        model = keras.Model(inp, out)
        y0 = model(sample)
        path = os.path.join(tmp_path, "fbs.keras")
        model.save(path)
        loaded = keras.models.load_model(
            path, custom_objects={"FrequencyBandStatistics": FrequencyBandStatistics}
        )
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(y0), keras.ops.convert_to_numpy(loaded(sample)),
            rtol=1e-5, atol=1e-5,
        )


class TestPRISMLayer:
    def _make(self, **kw):
        defaults = dict(tree_depth=2, num_wavelet_levels=2, router_hidden_dim=16,
                        dropout_rate=0.0)
        defaults.update(kw)
        return PRISMLayer(**defaults)

    def test_forward_pass(self, sample):
        out = self._make()(sample)
        assert tuple(out.shape) == (B, SEQ, CH)
        assert np.all(np.isfinite(keras.ops.convert_to_numpy(out)))

    def test_compute_output_shape(self):
        assert self._make().compute_output_shape((B, SEQ, CH)) == (B, SEQ, CH)

    def test_serialization_round_trip(self, sample, tmp_path):
        inp = keras.Input(shape=(SEQ, CH))
        out = self._make(name="prism")(inp)
        model = keras.Model(inp, out)
        y0 = model(sample, training=False)
        path = os.path.join(tmp_path, "prism.keras")
        model.save(path)
        loaded = keras.models.load_model(
            path, custom_objects={"PRISMLayer": PRISMLayer}
        )
        y1 = loaded(sample, training=False)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(y0), keras.ops.convert_to_numpy(y1),
            rtol=1e-5, atol=1e-5,
        )

    def test_get_config_round_trip(self):
        rebuilt = PRISMLayer.from_config(self._make(use_residual=False).get_config())
        assert rebuilt.use_residual is False
