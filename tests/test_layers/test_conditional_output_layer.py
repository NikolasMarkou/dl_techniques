"""Tests for the ConditionalOutputLayer (batch-wise tensor selector)."""

import os
import keras
import numpy as np
import pytest

from dl_techniques.layers.conditional_output_layer import ConditionalOutputLayer

B, D = 4, 5


class TestConditionalOutputLayer:

    def test_construction(self):
        layer = ConditionalOutputLayer()
        assert layer.supports_masking is True

    def test_selection_logic(self):
        """All-zero ground-truth samples select inference; others keep gt."""
        gt = np.ones((B, D), dtype="float32")
        gt[1] = 0.0  # sample 1 is "unlabeled" (all zeros)
        inference = np.full((B, D), 9.0, dtype="float32")

        out = keras.ops.convert_to_numpy(
            ConditionalOutputLayer()([
                keras.ops.convert_to_tensor(gt),
                keras.ops.convert_to_tensor(inference),
            ])
        )
        # Sample 1 -> inference; others -> ground truth.
        np.testing.assert_allclose(out[1], inference[1])
        np.testing.assert_allclose(out[0], gt[0])

    def test_invalid_input_count_raises(self):
        layer = ConditionalOutputLayer()
        with pytest.raises(ValueError):
            layer([keras.ops.zeros((B, D))])

    def test_compute_output_shape(self):
        layer = ConditionalOutputLayer()
        assert layer.compute_output_shape([(B, D), (B, D)]) == (B, D)

    def test_serialization_round_trip(self, tmp_path):
        gt_in = keras.Input(shape=(D,), name="gt")
        inf_in = keras.Input(shape=(D,), name="inf")
        out = ConditionalOutputLayer(name="cond")([gt_in, inf_in])
        model = keras.Model([gt_in, inf_in], out)

        gt = np.random.default_rng(0).standard_normal((B, D)).astype("float32")
        inf = np.random.default_rng(1).standard_normal((B, D)).astype("float32")
        y0 = model([gt, inf])

        path = os.path.join(tmp_path, "cond.keras")
        model.save(path)
        loaded = keras.models.load_model(
            path, custom_objects={"ConditionalOutputLayer": ConditionalOutputLayer}
        )
        y1 = loaded([gt, inf])
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(y0), keras.ops.convert_to_numpy(y1), atol=1e-6
        )
