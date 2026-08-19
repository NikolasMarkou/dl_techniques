"""M2 .keras round-trip + validation tests for SwinTransformer.

Covers: construction via create_swin_transformer / from_variant, a ValueError
input-validation path (H4), forward pass, and a full save -> load ->
identical-output round-trip (atol 1e-5; GPU fp32 reduction noise).
"""

import os
import tempfile

import keras
import numpy as np
import pytest
from keras import ops

from dl_techniques.models.swin_transformer.model import (
    SwinTransformer,
    create_swin_transformer,
)


# [CORRECTED 2026-08-01, G-03] This used to say: "The 'tiny' variant uses
# window_size=7, so the feature grid at the deepest stage must be divisible by 7
# -> the smallest legal square input is 224x224." The second half is FALSE and is
# deleted, not softened. `window_size` constrains nothing: SwinTransformerBlock
# pads a short grid up to the window and crops back. Re-measured by execution --
# `create_swin_transformer("tiny", 10, input_shape=(s,s,3))` (window_size=7,
# patch_size=4) builds AND forwards finite logits at s = 224, 96, 64, 56 and 32.
# 224 is kept below because it is the variant's design resolution, not a minimum.
def _images(b=2, s=224, c=3):
    return np.random.rand(b, s, s, c).astype("float32")


class TestSwinValidation:

    def test_invalid_num_classes(self):
        with pytest.raises(ValueError, match="num_classes must be positive"):
            SwinTransformer(num_classes=0, input_shape=(224, 224, 3))

    def test_invalid_window_size(self):
        with pytest.raises(ValueError, match="window_size must be positive"):
            SwinTransformer(num_classes=10, window_size=0,
                            input_shape=(224, 224, 3))


class TestSwinRoundTrip:

    def test_forward_shape(self):
        model = create_swin_transformer("tiny", 10, input_shape=(224, 224, 3))
        out = model(_images(), training=False)
        assert out.shape == (2, 10)

    def test_keras_round_trip(self):
        model = create_swin_transformer("tiny", 10, input_shape=(224, 224, 3))
        x = _images()
        y0 = model(x, training=False)

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "swin.keras")
            model.save(path)
            reloaded = keras.models.load_model(path)
            y1 = reloaded(x, training=False)

        np.testing.assert_allclose(
            ops.convert_to_numpy(y0), ops.convert_to_numpy(y1),
            rtol=1e-5, atol=1e-5,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ---------------------------------------------------------------------
# Gradient flow (plan-2026-08-19-a616f581 step 10)
# ---------------------------------------------------------------------

from ..gradient_flow_oracle import assert_gradients_reach_every_trainable_weight


class TestSwinGradientFlow:
    """Every trainable weight must be on the backward graph.

    Run at 32x32 rather than this file's 224 default: the geometry note above
    establishes that 32 is legal, and the smaller grid keeps the tape step cheap
    (176 trainable weights either way). The interesting tensors here are the
    per-block ``relative_position_bias_table`` entries and the shifted-window
    attention projections -- a shift/roll implementation that dropped one of them
    would keep every shape and every round-trip test green.
    """

    def test_gradients_reach_every_trainable_weight(self):
        model = create_swin_transformer("tiny", 10, input_shape=(32, 32, 3))
        x = _images(s=32)
        model(x, training=False)  # a subclassed model is unbuilt until first call

        report = assert_gradients_reach_every_trainable_weight(model, x)

        assert len(report) == len(model.trainable_weights)
        assert len(report) > 0
        assert max(v for v in report.values() if v is not None) > 0.0
